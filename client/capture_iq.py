#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import socket
import sys
import time
from datetime import datetime

import numpy as np
import uhd
import yaml


# -----------------------------
# Logging (keep similar style)
# -----------------------------
class LogFormatter(logging.Formatter):
    @staticmethod
    def pp_now():
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)

    def formatTime(self, record, datefmt=None):
        if datefmt:
            return self.converter(record.created).strftime(datefmt)
        return LogFormatter.pp_now()


def build_logger(script_dir: str) -> logging.Logger:
    logger = logging.getLogger("rx_bin_capture")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(LogFormatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    fh = logging.FileHandler(os.path.join(script_dir, "rx_capture_bin.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(LogFormatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


def load_yaml_into_globals(script_dir: str, logger: logging.Logger) -> None:
    yml_path = os.path.join(script_dir, "cal-settings.yml")
    try:
        with open(yml_path, "r") as f:
            vars_ = yaml.safe_load(f)
        if not isinstance(vars_, dict):
            raise ValueError("cal-settings.yml content is not a dict")
        globals().update(vars_)
        logger.info("Loaded settings from %s", yml_path)
    except FileNotFoundError:
        logger.error("Cannot find %s", yml_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load %s: %s", yml_path, e)
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="RX: capture IQ and save to .bin")
    p.add_argument(
        "--out-dir",
        type=str,
        default="iq_captures",
        help="Output directory (default: iq_captures)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="File prefix (default auto timestamp)",
    )
    p.add_argument(
        "--channels",
        type=str,
        default="0,1",
        help="RX channels, e.g. '0' or '0,1' (default: 0,1)",
    )
    p.add_argument(
        "--fpga",
        type=str,
        default=None,
        help="Optional FPGA image path (e.g. usrp_b210_fpga_loopback.bin)",
    )
    p.add_argument(
        "--device-args",
        type=str,
        default="mode_n=integer",
        help="Extra device args for MultiUSRP() (default: mode_n=integer)",
    )
    return p.parse_args()


def setup_usrp_rx_only(usrp: uhd.usrp.MultiUSRP, channels, logger: logging.Logger):
    """
    RX-only minimal setup, reading variables from cal-settings.yml via globals():
      RATE, RX_GAIN, FREQ, etc.
    """
    # These must exist in your yaml or fallback here
    rate = float(globals().get("RATE", 250e3))
    rx_gain = float(globals().get("RX_GAIN", 22))
    freq = float(globals().get("FREQ", 0.0))

    # master clock if provided in your yaml
    mcr = float(globals().get("MASTER_CLOCK_RATE", 20e6))
    usrp.set_master_clock_rate(mcr)

    # optional clock/time sources from yaml (if you keep them there)
    clock_src = globals().get("CLOCK_SRC", None)     # e.g. "external"
    time_src = globals().get("TIME_SRC", None)       # e.g. "external"

    if clock_src:
        usrp.set_clock_source(clock_src)
        logger.info("Clock source: %s", clock_src)
    if time_src:
        usrp.set_time_source(time_src)
        logger.info("Time/PPS source: %s", time_src)

    # bandwidth from yaml if you want
    rx_bw = float(globals().get("RX_BW", 200e3))

    for ch in channels:
        usrp.set_rx_antenna("TX/RX", ch)   # ★ 强制使用 TX/RX 口
        usrp.set_rx_rate(rate, ch)
        usrp.set_rx_gain(rx_gain, ch)
        usrp.set_rx_bandwidth(rx_bw, ch)
        usrp.set_rx_dc_offset(True, ch)
        usrp.set_rx_agc(False, ch)

    # Tune if freq != 0 (按你原脚本语义：0 表示默认中心频率不动)
    if float(freq) != 0.0:
        treq = uhd.types.TuneRequest(float(freq))
        for ch in channels:
            res = usrp.set_rx_freq(treq, ch)
            logger.info(
                "CH%d RX tuned: target %.6f MHz, actual %.6f MHz",
                ch,
                res.target_rf_freq / 1e6,
                res.actual_rf_freq / 1e6,
            )
    else:
        logger.info("FREQ=0 -> skip tuning (keep current RX freq)")

    # Build RX streamer
    # CPU format: fc32 (np.complex64), wire format: sc16
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels
    rx_streamer = usrp.get_rx_stream(st_args)

    logger.info("RX setup done: rate=%.3f ksps, gain=%.1f dB, bw=%.1f kHz",
                rate / 1e3, rx_gain, rx_bw / 1e3)

    return rx_streamer


def capture_to_bin(
    usrp: uhd.usrp.MultiUSRP,
    rx_streamer: uhd.usrp.RXStreamer,
    channels,
    out_bin_path: str,
    logger: logging.Logger,
):
    rate = float(globals().get("RATE", 250e3))
    duration = float(globals().get("CAPTURE_TIME", 10.0))

    num_channels = rx_streamer.get_num_channels()
    max_samps_per_packet = rx_streamer.get_max_num_samps()

    rx_md = uhd.types.RXMetadata()
    recv_buffer = np.zeros((num_channels, max_samps_per_packet), dtype=np.complex64)

    # ---- start streaming with time alignment (multi-chan safe) ----
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False

    # 让开始时间稍微延后一点（比如 0.10s）
    start_delay = float(globals().get("RX_START_DELAY_S", 0.10))
    start_time = usrp.get_time_now().get_real_secs() + start_delay
    stream_cmd.time_spec = uhd.types.TimeSpec(start_time)

    rx_streamer.issue_stream_cmd(stream_cmd)
    logger.info("Scheduled RX start at device time %.6f (delay %.3f s)", start_time, start_delay)
    t0 = time.time()

    # stats
    total_samps = 0
    overflow_cnt = 0
    other_err_cnt = 0

    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    with open(out_bin_path, "wb") as f:
        try:
            while True:
                if (time.time() - t0) >= duration:
                    break

                n = rx_streamer.recv(recv_buffer, rx_md, timeout=1.0)

                if rx_md.error_code == uhd.types.RXMetadataErrorCode.none:
                    if n > 0:
                        # write (num_channels, n) complex64 to bin
                        f.write(recv_buffer[:, :n].tobytes(order="C"))
                        total_samps += n

                elif rx_md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    overflow_cnt += 1
                    if overflow_cnt <= 5 or overflow_cnt % 200 == 0:
                        logger.warning("RX overflow (%d)", overflow_cnt)
                else:
                    other_err_cnt += 1
                    logger.error("RX error: %s", str(rx_md.error_code))
                    if other_err_cnt > 20:
                        logger.error("Too many RX errors, stop.")
                        break

        except KeyboardInterrupt:
            logger.info("CTRL+C detected, stop early.")
        finally:
            rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    elapsed = time.time() - t0
    logger.info("Done. wrote %d samples/channel (approx %.3f s), elapsed=%.3f s",
                total_samps, total_samps / rate, elapsed)

    meta = {
        "host": socket.gethostname(),
        "channels": [int(c) for c in channels],
        "num_channels": int(num_channels),
        "bin_path": os.path.basename(out_bin_path),
        "dtype": "complex64 (I=float32, Q=float32)",
        "layout": "written as (num_channels, n) blocks in C-order, repeated over time",
        "rate_sps": float(rate),
        "capture_time_s": float(duration),
        "samples_per_channel": int(total_samps),
        "overflow_count": int(overflow_cnt),
        "other_error_count": int(other_err_cnt),
        "device_time_end": float(usrp.get_time_now().get_real_secs()),
        "wallclock_end": datetime.now().isoformat(),
    }
    return meta


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    logger = build_logger(script_dir)

    # load yaml like your original script
    load_yaml_into_globals(script_dir, logger)

    # channels
    channels = [int(x.strip()) for x in args.channels.split(",") if x.strip() != ""]
    if not channels:
        logger.error("No channels specified.")
        sys.exit(1)

    # output naming
    os.makedirs(args.out_dir, exist_ok=True)
    if args.prefix is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"iq_{socket.gethostname()}_{ts}"
    else:
        prefix = args.prefix

    out_bin = os.path.join(args.out_dir, f"{prefix}.bin")
    out_yml = os.path.join(args.out_dir, f"{prefix}.yml")

    # build device args
    dev_addr = args.device_args.strip()
    if args.fpga:
        dev_addr = f"fpga={os.path.abspath(args.fpga)}, {dev_addr}"

    logger.info("Device args: %s", dev_addr)
    usrp = uhd.usrp.MultiUSRP(dev_addr)
    logger.info("Using Device:\n%s", usrp.get_pp_string())

    rx_streamer = setup_usrp_rx_only(usrp, channels, logger)

    meta = capture_to_bin(usrp, rx_streamer, channels, out_bin, logger)

    # also dump ALL loaded yaml parameters into the same meta (for reproducibility)
    # 注意：globals() 很大，这里只挑你 yaml 里那些键（我们在 load 时拿到的 vars_ 没保存下来）
    # 简化做法：再读一遍 yaml
    try:
        with open(os.path.join(script_dir, "cal-settings.yml"), "r") as f:
            vars_ = yaml.safe_load(f) or {}
        meta["cal_settings"] = vars_
    except Exception:
        meta["cal_settings"] = None

    with open(out_yml, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    logger.info("Saved BIN: %s", out_bin)
    logger.info("Saved META: %s", out_yml)


if __name__ == "__main__":
    main()
