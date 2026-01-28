#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import json
import numpy as np
import zmq


# =========================
# Helpers: load sequences
# =========================
def load_tx_seq(path: str, L: int) -> np.ndarray:
    s = np.fromfile(path, dtype=np.complex64)
    if s.size < L:
        raise ValueError(f"{path}: length {s.size} < L={L}")
    s = s[:L].astype(np.complex64, copy=False)

    # Unit RMS normalization (robust)
    s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)
    return s


# =========================
# Alignment + correlation
# =========================
def find_best_cyclic_shift(y: np.ndarray, s: np.ndarray, max_search: int = 512) -> int:
    """
    Find shift k in [0, max_search) maximizing |<roll(s,k), y>|.
    Assumes y,s length L.
    """
    best_k, best_v = 0, -1.0
    for k in range(max_search):
        v = np.abs(np.vdot(np.roll(s, k), y))
        if v > best_v:
            best_v, best_k = v, k
    return best_k


def corr_leakage_sinr(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    user_id: int,
    max_shift_search: int = 512,
    eps: float = 1e-12,
) -> dict:
    """
    Compute:
      a_des = <s_des, y>/L
      a_int = <s_int, y>/L
      leak_db = 10log10(|a_int|^2 / |a_des|^2)
      residual noise power: mean(|y - a_des*s_des - a_int*s_int|^2)
      SINR = |a_des|^2 / (|a_int|^2 + P_noise)
    """
    L = len(s1)
    if len(y) != L or len(s2) != L:
        raise ValueError("Length mismatch in corr_leakage_sinr()")

    if user_id not in (1, 2):
        raise ValueError("user_id must be 1 or 2")

    s_des = s1 if user_id == 1 else s2
    s_int = s2 if user_id == 1 else s1

    # alignment (cyclic shift)
    shift = find_best_cyclic_shift(y, s_des, max_search=max_shift_search)
    s_des_s = np.roll(s_des, shift)
    s_int_s = np.roll(s_int, shift)

    # projections
    a_des = np.vdot(s_des_s, y) / float(L)
    a_int = np.vdot(s_int_s, y) / float(L)

    P_des = float(np.abs(a_des) ** 2)
    P_int = float(np.abs(a_int) ** 2)

    leak_ratio = P_int / max(P_des, eps)
    leak_db = 10.0 * np.log10(max(leak_ratio, eps))

    # residual-based noise/model error estimate
    e = y - a_des * s_des_s - a_int * s_int_s
    P_noise = float(np.mean(np.abs(e) ** 2))

    sinr = P_des / max(P_int + P_noise, eps)
    sinr_db = 10.0 * np.log10(max(sinr, eps))

    return {
        "shift": int(shift),
        "a_des_re": float(np.real(a_des)),
        "a_des_im": float(np.imag(a_des)),
        "a_int_re": float(np.real(a_int)),
        "a_int_im": float(np.imag(a_int)),
        "P_des": float(P_des),
        "P_int": float(P_int),
        "P_noise": float(P_noise),
        "leak_ratio": float(leak_ratio),
        "leak_db": float(leak_db),
        "sinr": float(sinr),
        "sinr_db": float(sinr_db),
    }


# =========================
# ZMQ message parsing
# =========================
def parse_iq_message(parts):
    """
    Try to parse IQ from different ZMQ payload styles.

    Supported (best-effort):
      1) multipart: [topic, raw_bytes] where raw_bytes is complex64 array
      2) single frame raw_bytes complex64
      3) single frame JSON string with {"iq": [...]}  (not recommended, but supported)

    Returns: (topic_str, np.ndarray complex64)
    """
    topic = ""
    payload = None

    if len(parts) == 2:
        # typical PUB/SUB style: topic + raw
        topic = parts[0].decode(errors="ignore")
        payload = parts[1]
    elif len(parts) == 1:
        payload = parts[0]
    else:
        raise ValueError(f"Unexpected ZMQ multipart length: {len(parts)}")

    # If payload looks like JSON (text), try JSON path
    if isinstance(payload, (bytes, bytearray)) and len(payload) > 0:
        b0 = payload[:1]
        if b0 in (b"{", b"["):
            try:
                s = payload.decode("utf-8", errors="strict")
                d = json.loads(s)
                # expects iq as list of [re, im] or complex pairs; but this is rare
                iq = d.get("iq", None)
                if iq is None:
                    raise ValueError("JSON has no 'iq' field")
                # accept list of [re,im]
                arr = np.array([c[0] + 1j * c[1] for c in iq], dtype=np.complex64)
                return topic, arr
            except Exception:
                # fall back to raw-bytes parsing
                pass

    # Raw complex64 bytes
    arr = np.frombuffer(payload, dtype=np.complex64)
    if arr.size == 0:
        raise ValueError("Empty IQ payload")
    return topic, arr


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="UE correlation despreading + leakage/SINR publisher (ZMQ).")
    ap.add_argument("--ap-ip", type=str, default="127.0.0.1",
                    help="IP of the IQ publisher (AP). If running on the AP itself, keep default 127.0.0.1.")
    ap.add_argument("--ap-port", type=int, default=50001, help="IQ PUB port on AP (default 50001).")
    ap.add_argument("--topic", type=str, default="CH0",
                    help="ZMQ topic to subscribe (e.g., CH0/CH1). Use empty string to subscribe all.")
    ap.add_argument("--user-id", type=int, default=1, choices=[1, 2], help="UE id: 1 => desired tx1, 2 => desired tx2.")
    ap.add_argument("--L", type=int, default=4096, help="Correlation window length (default 4096).")
    ap.add_argument("--max-shift-search", type=int, default=512, help="Cyclic shift search range (default 512).")
    ap.add_argument("--tx1", type=str, default="tx1.bin", help="Path to tx1.bin")
    ap.add_argument("--tx2", type=str, default="tx2.bin", help="Path to tx2.bin")
    ap.add_argument("--pub-port", type=int, default=52001, help="Local PUB port for server to subscribe (default 52001).")
    ap.add_argument("--rate-hz", type=float, default=20.0,
                    help="Max publish rate (Hz). We still process all IQ, but only publish up to this rate.")
    args = ap.parse_args()

    # Resolve paths relative to script dir (convenient)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    tx1_path = args.tx1 if os.path.isabs(args.tx1) else os.path.join(script_dir, args.tx1)
    tx2_path = args.tx2 if os.path.isabs(args.tx2) else os.path.join(script_dir, args.tx2)

    L = int(args.L)
    s1 = load_tx_seq(tx1_path, L)
    s2 = load_tx_seq(tx2_path, L)

    # ZMQ setup
    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.ap_ip}:{args.ap_port}")
    topic_bytes = args.topic.encode() if args.topic is not None else b""
    sub.setsockopt(zmq.SUBSCRIBE, topic_bytes)  # b"" subscribes all

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.pub_port}")

    # Buffers
    buf = np.zeros(L, dtype=np.complex64)
    fill = 0

    # Publish throttling
    min_pub_dt = 1.0 / max(float(args.rate_hz), 1e-6)
    last_pub_t = 0.0

    print(f"[UE] Subscribe IQ from tcp://{args.ap_ip}:{args.ap_port} topic={args.topic!r}")
    print(f"[UE] Publish metrics on tcp://*:{args.pub_port} (server will read 'evm_pct')")
    print(f"[UE] user_id={args.user_id}, L={L}, max_shift_search={args.max_shift_search}")
    print(f"[UE] tx1={tx1_path}")
    print(f"[UE] tx2={tx2_path}")

    # PUB/SUB warm-up
    time.sleep(0.2)

    try:
        while True:
            parts = sub.recv_multipart()
            topic, x = parse_iq_message(parts)

            # Append into L window(s)
            i = 0
            while i < x.size:
                take = min(L - fill, x.size - i)
                buf[fill:fill + take] = x[i:i + take]
                fill += take
                i += take

                if fill == L:
                    y = buf.copy()
                    fill = 0

                    metrics = corr_leakage_sinr(
                        y=y,
                        s1=s1,
                        s2=s2,
                        user_id=args.user_id,
                        max_shift_search=int(args.max_shift_search),
                    )

                    now = time.time()
                    if (now - last_pub_t) >= min_pub_dt:
                        # IMPORTANT: keep 'evm_pct' for server compatibility
                        msg = {
                            "t": now,
                            "evm_pct": float("nan"),   # server expects this key; we don't compute EVM here
                            "user_id": int(args.user_id),
                            "topic": topic,
                            "L": L,
                            "metrics": metrics,
                        }
                        pub.send_string(json.dumps(msg, allow_nan=True))
                        last_pub_t = now

    except KeyboardInterrupt:
        print("\n[UE] Ctrl+C received, exiting.")

    finally:
        try:
            sub.close(0)
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass
        # ctx.term() 不要强制 term（Context.instance 可能被别处复用）
        sys.exit(0)


if __name__ == "__main__":
    main()
