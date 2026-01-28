#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import json
import threading
import numpy as np
import zmq

import uhd


# ---------- tx sequence helpers ----------
def load_tx_seq(path: str, L: int) -> np.ndarray:
    s = np.fromfile(path, dtype=np.complex64)
    if s.size < L:
        raise ValueError(f"{path}: length {s.size} < L={L}")
    s = s[:L].astype(np.complex64, copy=False)
    s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)  # unit RMS
    return s


def find_best_cyclic_shift(y: np.ndarray, s: np.ndarray, max_search: int = 512) -> int:
    best_k, best_v = 0, -1.0
    for k in range(max_search):
        v = np.abs(np.vdot(np.roll(s, k), y))
        if v > best_v:
            best_v, best_k = v, k
    return best_k


def corr_leakage_sinr(y, s1, s2, user_id, max_shift_search=512, eps=1e-12):
    L = len(s1)
    s_des = s1 if user_id == 1 else s2
    s_int = s2 if user_id == 1 else s1

    shift = find_best_cyclic_shift(y, s_des, max_search=max_shift_search)
    s_des_s = np.roll(s_des, shift)
    s_int_s = np.roll(s_int, shift)

    a_des = np.vdot(s_des_s, y) / float(L)
    a_int = np.vdot(s_int_s, y) / float(L)

    P_des = float(np.abs(a_des) ** 2)
    P_int = float(np.abs(a_int) ** 2)
    leak_ratio = P_int / max(P_des, eps)
    leak_db = 10.0 * np.log10(max(leak_ratio, eps))

    e = y - a_des * s_des_s - a_int * s_int_s
    P_noise = float(np.mean(np.abs(e) ** 2))

    sinr = P_des / max(P_int + P_noise, eps)
    sinr_db = 10.0 * np.log10(max(sinr, eps))

    return {
        "shift": int(shift),
        "P_des": P_des,
        "P_int": P_int,
        "P_noise": P_noise,
        "leak_ratio": float(leak_ratio),
        "leak_db": float(leak_db),
        "sinr": float(sinr),
        "sinr_db": float(sinr_db),
        "a_des_re": float(np.real(a_des)),
        "a_des_im": float(np.imag(a_des)),
        "a_int_re": float(np.real(a_int)),
        "a_int_im": float(np.imag(a_int)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", type=str, default="", help="USRP serial (optional).")
    ap.add_argument("--freq", type=float, default=920e6)
    ap.add_argument("--rate", type=float, default=250e3)
    ap.add_argument("--gain", type=float, default=40.0)
    ap.add_argument("--antenna", type=str, default="TX/RX")
    ap.add_argument("--user-id", type=int, choices=[1, 2], required=True)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--max-shift-search", type=int, default=512)
    ap.add_argument("--tx1", type=str, default="tx1.bin")
    ap.add_argument("--tx2", type=str, default="tx2.bin")
    ap.add_argument("--pub-port", type=int, default=52001)
    ap.add_argument("--publish-hz", type=float, default=10.0)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    tx1_path = args.tx1 if os.path.isabs(args.tx1) else os.path.join(script_dir, args.tx1)
    tx2_path = args.tx2 if os.path.isabs(args.tx2) else os.path.join(script_dir, args.tx2)

    L = int(args.L)
    s1 = load_tx_seq(tx1_path, L)
    s2 = load_tx_seq(tx2_path, L)

    # ---- ZMQ PUB (to server) ----
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.pub_port}")
    time.sleep(0.2)  # warm-up

    # ---- UHD RX ----
    dev_args = ""
    if args.serial:
        dev_args = f"serial={args.serial}"
    usrp = uhd.usrp.MultiUSRP(dev_args)
    usrp.set_rx_rate(args.rate, 0)
    usrp.set_rx_freq(args.freq, 0)
    usrp.set_rx_gain(args.gain, 0)
    usrp.set_rx_antenna(args.antenna, 0)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)

    max_samps = rx_streamer.get_max_num_samps()
    recv_buf = np.zeros((1, max_samps), dtype=np.complex64)
    md = uhd.types.RXMetadata()

    # stream start
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    # window buffer
    win = np.zeros(L, dtype=np.complex64)
    fill = 0

    min_pub_dt = 1.0 / max(args.publish_hz, 1e-6)
    last_pub = 0.0
    last_print = 0.0

    print(f"[RPI RX] USRP rate={args.rate} freq={args.freq} gain={args.gain} L={L} user_id={args.user_id}")
    print(f"[RPI RX] Publishing results on tcp://*:{args.pub_port} (server subscribes rpi-t03.local:52001)")

    try:
        while True:
            n = rx_streamer.recv(recv_buf, md, timeout=1.0)
            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                # just continue; keep script alive
                continue
            if n <= 0:
                continue

            x = recv_buf[0, :n]

            i = 0
            while i < x.size:
                take = min(L - fill, x.size - i)
                win[fill:fill + take] = x[i:i + take]
                fill += take
                i += take

                if fill == L:
                    y = win.copy()
                    fill = 0

                    metrics = corr_leakage_sinr(
                        y=y, s1=s1, s2=s2,
                        user_id=args.user_id,
                        max_shift_search=args.max_shift_search,
                    )

                    now = time.time()
                    if now - last_pub >= min_pub_dt:
                        # keep 'evm_pct' key for your existing server script
                        msg = {
                            "t": now,
                            "evm_pct": float("nan"),     # server expects this key
                            "user_id": int(args.user_id),
                            "metrics": metrics,
                        }
                        pub.send_string(json.dumps(msg, allow_nan=True))
                        last_pub = now

                    # local console heartbeat (1 Hz)
                    if now - last_print >= 1.0:
                        print(f"sinr_db={metrics['sinr_db']:.1f}  leak_db={metrics['leak_db']:.1f}  shift={metrics['shift']}",
                              flush=True)
                        last_print = now

    except KeyboardInterrupt:
        print("\n[RPI RX] Ctrl+C, stopping...")

    finally:
        # stream stop
        cmd2 = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(cmd2)
        try:
            pub.close(0)
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
