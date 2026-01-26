#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import socket
import sys
import threading
import time
from datetime import datetime, timedelta
import queue
import json

import numpy as np
import uhd
import yaml
import zmq

import tools  # 你自己的工具模块


# =============================================================================
#                           Experiment Configuration
# =============================================================================
CMD_DELAY = 0.05
RX_TX_SAME_CHANNEL = True
CLOCK_TIMEOUT = 1000
INIT_DELAY = 0.2
RATE = 250e3
LOOPBACK_TX_GAIN = 50
RX_GAIN = 22
CAPTURE_TIME = 10
FREQ = 0

meas_id = 0
exp_id = 0

SWITCH_LOOPBACK_MODE = 0x00000006
SWITCH_RESET_MODE = 0x00000000

# =============================================================================
#                   1-bit DAC Quantization (optional)
# =============================================================================
ENABLE_1BIT_DAC = True
ENABLE_DITHER = False
DITHER_REL_STD = 0.10
ONEBIT_POWER_NORM = True
DITHER_SEED_BASE = 12345


# =============================================================================
#                           ZMQ
# =============================================================================
context = zmq.Context()

HOSTNAME = socket.gethostname()[4:]  # 你原来就是这样截的
file_open = False

# =============================================================================
#                           Logger
# =============================================================================
class LogFormatter(logging.Formatter):
    @staticmethod
    def pp_now():
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)

    def formatTime(self, record, datefmt=None):
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date


class ColoredFormatter(LogFormatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def fmt(val):
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


DEG = "\u00b0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
logger.addHandler(console)

formatter = LogFormatter(fmt="[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s")
console.setFormatter(ColoredFormatter(fmt=formatter._fmt))

file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "log.txt"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

begin_time = 2.0


# =============================================================================
#                           Channel mapping
# =============================================================================
if RX_TX_SAME_CHANNEL:
    REF_RX_CH = FREE_TX_CH = 0
    LOOPBACK_RX_CH = LOOPBACK_TX_CH = 1
    logger.debug("\nPLL REF → CH0 RX\nCH1 TX → CH1 RX\nCH0 TX →")
else:
    LOOPBACK_RX_CH = FREE_TX_CH = 0
    REF_RX_CH = LOOPBACK_TX_CH = 1
    logger.debug("\nPLL REF → CH1 RX\nCH1 TX → CH0 RX\nCH0 TX →")


# =============================================================================
#                   Dither + 1-bit Quantization helpers
# =============================================================================
def add_complex_dither(x: np.ndarray, rel_std: float, rng: np.random.Generator) -> np.ndarray:
    if rel_std <= 0 or x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if rms <= 0:
        return x
    sigma = rel_std * rms
    d = (rng.normal(0.0, sigma / np.sqrt(2), size=x.shape) +
         1j * rng.normal(0.0, sigma / np.sqrt(2), size=x.shape)).astype(np.complex64)
    return x + d


def one_bit_quantize_complex(x: np.ndarray, power_norm: bool = True, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.complex64)
    re = np.real(x)
    im = np.imag(x)
    re_q = np.where(re >= 0, 1.0, -1.0)
    im_q = np.where(im >= 0, 1.0, -1.0)
    q = (re_q + 1j * im_q).astype(np.complex64)

    if not power_norm:
        return q

    p_in = float(np.mean(np.abs(x) ** 2))
    p_q = float(np.mean(np.abs(q) ** 2))
    scale = np.sqrt(max(p_in, eps) / max(p_q, eps))
    return (scale * q).astype(np.complex64)


# =============================================================================
#                       TX.bin loader (complex64 ONLY)
# =============================================================================
def load_tx_bin_complex64(bin_path: str) -> np.ndarray:
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"tx waveform file not found: {bin_path}")
    x = np.fromfile(bin_path, dtype=np.complex64)
    if x.size == 0:
        raise ValueError(f"tx.bin contains 0 samples: {bin_path}")
    return x.astype(np.complex64, copy=False)


# =============================================================================
#                           RX: reference capture + phase estimation
# =============================================================================
def rx_ref(usrp, rx_streamer, quit_event, duration, result_queue, start_time=None):
    logger.debug(f"GAIN IS CH0: {usrp.get_rx_gain(0)} CH1: {usrp.get_rx_gain(1)}")

    num_channels = rx_streamer.get_num_channels()
    max_samps_per_packet = rx_streamer.get_max_num_samps()
    buffer_length = int(duration * RATE * 2)
    iq_data = np.empty((num_channels, buffer_length), dtype=np.complex64)

    recv_buffer = np.zeros((num_channels, max_samps_per_packet), dtype=np.complex64)
    rx_md = uhd.types.RXMetadata()

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    timeout = 1.0

    if start_time is not None:
        stream_cmd.time_spec = start_time
        time_diff = start_time.get_real_secs() - usrp.get_time_now().get_real_secs()
        if time_diff > 0:
            timeout = 1.0 + time_diff
    else:
        stream_cmd.time_spec = uhd.types.TimeSpec(
            usrp.get_time_now().get_real_secs() + INIT_DELAY + 0.1
        )

    rx_streamer.issue_stream_cmd(stream_cmd)

    try:
        num_rx = 0
        while not quit_event.is_set():
            num_rx_i = rx_streamer.recv(recv_buffer, rx_md, timeout)
            if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                logger.error(rx_md.error_code)
                continue
            if num_rx_i <= 0:
                continue

            samples = recv_buffer[:, :num_rx_i]
            if num_rx + num_rx_i > buffer_length:
                logger.error("more samples received than buffer long, not storing the data")
            else:
                iq_data[:, num_rx: num_rx + num_rx_i] = samples
                num_rx += num_rx_i

    except KeyboardInterrupt:
        pass

    finally:
        logger.debug("CTRL+C is pressed or duration is reached, closing off ")
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

        iq_samples = iq_data[:, int(RATE * 1): num_rx]  # drop first 1s

        phase_ch0, freq_slope_ch0_before, freq_slope_ch0_after = tools.get_phases_and_apply_bandpass(
            iq_samples[0, :]
        )
        phase_ch1, freq_slope_ch1_before, freq_slope_ch1_after = tools.get_phases_and_apply_bandpass(
            iq_samples[1, :]
        )

        logger.debug(
            "Frequency offset CH0: %.2f Hz -> %.2f Hz",
            float(freq_slope_ch0_before), float(freq_slope_ch0_after),
        )
        logger.debug(
            "Frequency offset CH1: %.2f Hz -> %.2f Hz",
            float(freq_slope_ch1_before), float(freq_slope_ch1_after),
        )

        phase_diff = tools.to_min_pi_plus_pi(phase_ch0 - phase_ch1, deg=False)
        _circ_mean = tools.circmean(phase_diff, deg=False)

        A_rms = np.sqrt(np.mean(np.abs(iq_samples) ** 2, axis=1))
        result_queue.put((A_rms[1], _circ_mean))


# =============================================================================
#                           USRP Clock/PPS/Tune
# =============================================================================
def setup_clock(usrp, clock_src, num_mboards):
    usrp.set_clock_source(clock_src)
    logger.debug("Now confirming lock on clock signals...")
    end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)
    for i in range(num_mboards):
        is_locked = usrp.get_mboard_sensor("ref_locked", i)
        while (not is_locked) and (datetime.now() < end_time):
            time.sleep(1e-3)
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
        if not is_locked:
            logger.error("Unable to confirm clock signal locked on board %d", i)
            return False
        logger.debug("Clock signals are locked")
    return True


def setup_pps(usrp, pps):
    logger.debug("Setting PPS")
    usrp.set_time_source(pps)
    return True


def print_tune_result(tune_res):
    logger.debug(
        "Tune Result:\n    Target RF  Freq: %s MHz\n    Actual RF  Freq: %s MHz\n    Target DSP Freq: %s MHz\n    Actual DSP Freq: %s MHz",
        fmt(tune_res.target_rf_freq / 1e6),
        fmt(tune_res.actual_rf_freq / 1e6),
        fmt(tune_res.target_dsp_freq / 1e6),
        fmt(tune_res.actual_dsp_freq / 1e6),
    )


def tune_usrp(usrp, freq, channels, at_time):
    treq = uhd.types.TuneRequest(freq)
    usrp.set_command_time(uhd.types.TimeSpec(at_time))
    treq.dsp_freq = 0.0
    treq.target_freq = freq
    treq.rf_freq = freq
    treq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    treq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    treq.args = uhd.types.DeviceAddr("mode_n=integer")

    rx_freq = freq - 1e3
    rreq = uhd.types.TuneRequest(rx_freq)
    rreq.rf_freq = rx_freq
    rreq.target_freq = rx_freq
    rreq.dsp_freq = 0.0
    rreq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    rreq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    rreq.args = uhd.types.DeviceAddr("mode_n=fractional")

    for chan in channels:
        logger.debug("RX tuning...")
        print_tune_result(usrp.set_rx_freq(rreq, chan))
        logger.debug("TX tuning...")
        print_tune_result(usrp.set_tx_freq(treq, chan))

    while not usrp.get_rx_sensor("lo_locked").to_bool():
        time.sleep(0.01)
    logger.info("RX LO is locked")

    while not usrp.get_tx_sensor("lo_locked").to_bool():
        time.sleep(0.01)
    logger.info("TX LO is locked")


# =============================================================================
#                           Server Sync
# =============================================================================
def wait_till_go_from_server(ip):
    global meas_id, file_open, data_file, file_name

    logger.debug("Connecting to server %s.", ip)
    sync_socket = context.socket(zmq.SUB)
    alive_socket = context.socket(zmq.REQ)

    sync_socket.connect(f"tcp://{ip}:{5557}")
    alive_socket.connect(f"tcp://{ip}:{5558}")
    sync_socket.subscribe("")

    logger.debug("Sending ALIVE")
    alive_socket.send_string(HOSTNAME)

    logger.debug("Waiting on SYNC from server %s.", ip)
    _meas_id_str, unique_id = sync_socket.recv_string().split(" ")
    meas_id = int(_meas_id_str)

    file_name = f"data_{HOSTNAME}_{unique_id}_{meas_id}"

    if not file_open:
        data_file = open(f"data_{HOSTNAME}_{unique_id}.txt", "a")
        file_open = True

    logger.debug("meas_id=%s", meas_id)

    alive_socket.close()
    sync_socket.close()


def send_usrp_in_tx_mode(ip):
    tx_mode_socket = context.socket(zmq.REQ)
    tx_mode_socket.connect(f"tcp://{ip}:{5559}")
    logger.debug("USRP IN TX MODE")
    tx_mode_socket.send_string(HOSTNAME)
    tx_mode_socket.close()


def setup(usrp, SERVER_IP):
    rate = RATE
    mcr = 20e6
    assert (mcr / rate).is_integer(), (
        f"The masterclock rate {mcr} should be an integer multiple of the sampling rate {rate}"
    )
    usrp.set_master_clock_rate(mcr)

    channels = [0, 1]
    setup_clock(usrp, "external", usrp.get_num_mboards())
    setup_pps(usrp, "external")

    rx_bw = 200e3
    for chan in channels:
        usrp.set_rx_rate(rate, chan)
        usrp.set_tx_rate(rate, chan)
        usrp.set_rx_dc_offset(True, chan)
        usrp.set_rx_bandwidth(rx_bw, chan)
        usrp.set_rx_agc(False, chan)

    # will be overridden by cal-settings.yml
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

    usrp.set_rx_gain(LOOPBACK_RX_GAIN, LOOPBACK_RX_CH)
    usrp.set_rx_gain(REF_RX_GAIN, REF_RX_CH)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels

    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)

    wait_till_go_from_server(SERVER_IP)

    logger.info("Setting device timestamp to 0...")
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    logger.debug("[SYNC] Resetting time.")

    time.sleep(2)
    tune_usrp(usrp, FREQ, channels, at_time=begin_time)
    logger.info(f"USRP has been tuned and setup. ({usrp.get_time_now().get_real_secs()})")

    return tx_streamer, rx_streamer


# =============================================================================
#                           Thread wrappers
# =============================================================================
def rx_thread(usrp, rx_streamer, quit_event, duration, res, start_time=None):
    _rx_thread = threading.Thread(
        target=rx_ref,
        args=(usrp, rx_streamer, quit_event, duration, res, start_time),
    )
    _rx_thread.name = "RX_thread"
    _rx_thread.start()
    return _rx_thread


def tx_async_th(tx_streamer, quit_event):
    async_metadata = uhd.types.TXAsyncMetadata()
    try:
        while not quit_event.is_set():
            if not tx_streamer.recv_async_msg(async_metadata, 0.01):
                continue
            if async_metadata.event_code != uhd.types.TXMetadataEventCode.burst_ack:
                logger.error(async_metadata.event_code)
    except KeyboardInterrupt:
        pass


def delta(usrp, at_time):
    return at_time - usrp.get_time_now().get_real_secs()


def starting_in(usrp, at_time):
    return f"Starting in {delta(usrp, at_time):.2f}s"


def tx_meta_thread(tx_streamer, quit_event):
    tx_meta_thr = threading.Thread(target=tx_async_th, args=(tx_streamer, quit_event))
    tx_meta_thr.name = "TX_META_thread"
    tx_meta_thr.start()
    return tx_meta_thr


# =============================================================================
#                           TX: 2-user waveform superposition (phase-only weights)
# =============================================================================
def _next_chunk(sig: np.ndarray, idx: int, n: int):
    if idx + n <= sig.size:
        out = sig[idx:idx + n]
        idx += n
        if idx >= sig.size:
            idx = 0
        return out, idx
    first = sig[idx:]
    remain = n - first.size
    second = sig[:remain]
    out = np.concatenate([first, second])
    idx = remain
    return out, idx


def tx_ref_mixed(
    usrp,
    tx_streamer,
    quit_event,
    start_time: uhd.types.TimeSpec,
    base_phase_common: float,
    phi_u1: float,
    phi_u2: float,
    tx_u1_path: str,
    tx_u2_path: str,
    active_tx_ch: int,
    active_tx_amp: float,
):
    """
    一轮同时发送两路波形（phase-only weight）：
        x(t) = e^{j(base_phase_common + phi_u1)} s1(t) + e^{j(base_phase_common + phi_u2)} s2(t)

    只在 active_tx_ch 那一路发（另一通道置零），与你原来的 tx_phase_coh 一致。
    """
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()

    s1 = load_tx_bin_complex64(tx_u1_path)
    s2 = load_tx_bin_complex64(tx_u2_path)
    logger.info("Loaded U1=%s (N=%d), U2=%s (N=%d)", tx_u1_path, s1.size, tx_u2_path, s2.size)

    w1 = np.exp(1j * (base_phase_common + phi_u1)).astype(np.complex64)
    w2 = np.exp(1j * (base_phase_common + phi_u2)).astype(np.complex64)

    # RNG
    try:
        seed = int(DITHER_SEED_BASE + int(meas_id))
    except Exception:
        seed = int(DITHER_SEED_BASE)
    rng = np.random.default_rng(seed)

    tx_md = uhd.types.TXMetadata()
    tx_md.time_spec = start_time
    tx_md.has_time_spec = True
    tx_md.end_of_burst = False

    idx1 = 0
    idx2 = 0

    try:
        while not quit_event.is_set():
            n = int(max_samps_per_packet)
            if n <= 0:
                continue

            c1, idx1 = _next_chunk(s1, idx1, n)
            c2, idx2 = _next_chunk(s2, idx2, n)

            # MU superposition
            x_base = (w1 * c1 + w2 * c2).astype(np.complex64, copy=False)

            transmit_buffer = np.zeros((num_channels, n), dtype=np.complex64)
            transmit_buffer[active_tx_ch, :] = (active_tx_amp * x_base).astype(np.complex64, copy=False)

            if ENABLE_DITHER:
                transmit_buffer[active_tx_ch, :] = add_complex_dither(
                    transmit_buffer[active_tx_ch, :], rel_std=DITHER_REL_STD, rng=rng
                )
            if ENABLE_1BIT_DAC:
                transmit_buffer[active_tx_ch, :] = one_bit_quantize_complex(
                    transmit_buffer[active_tx_ch, :], power_norm=ONEBIT_POWER_NORM
                )

            tx_streamer.send(transmit_buffer, tx_md)
            tx_md.has_time_spec = False

    finally:
        tx_md.end_of_burst = True
        tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), tx_md)


# =============================================================================
#                           Measurements
# =============================================================================
def measure_pilot(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time):
    logger.debug("########### Measure PILOT ###########")
    usrp.set_rx_antenna("TX/RX", 1)
    start_time = uhd.types.TimeSpec(at_time)
    logger.debug(starting_in(usrp, at_time))

    rx_thr = rx_thread(
        usrp, rx_streamer, quit_event,
        duration=CAPTURE_TIME, res=result_queue, start_time=start_time
    )

    time.sleep(CAPTURE_TIME + delta(usrp, at_time))
    quit_event.set()
    rx_thr.join()

    usrp.set_rx_antenna("RX2", 1)
    quit_event.clear()


def measure_loopback(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time):
    logger.debug("########### Measure LOOPBACK ###########")

    # 只开 LOOPBACK_TX_CH 这一路
    amplitudes = [0.0, 0.0]
    amplitudes[LOOPBACK_TX_CH] = 0.8

    start_time = uhd.types.TimeSpec(at_time)
    logger.debug(starting_in(usrp, at_time))

    user_settings = None
    try:
        user_settings = usrp.get_user_settings_iface(1)
        if user_settings:
            user_settings.poke32(0, SWITCH_LOOPBACK_MODE)
        else:
            logger.error("Cannot write to user settings.")
    except Exception as e:
        logger.error(e)

    # 用最简单的“发常数”方式做 loopback：这里复用你的单音逻辑
    tx_md = uhd.types.TXMetadata()
    tx_md.time_spec = start_time
    tx_md.has_time_spec = True
    tx_md.end_of_burst = False

    num_channels = tx_streamer.get_num_channels()
    max_samps = tx_streamer.get_max_num_samps()

    quit_event.clear()

    def _tx_loop():
        try:
            while not quit_event.is_set():
                buf = np.zeros((num_channels, max_samps), dtype=np.complex64)
                buf[LOOPBACK_TX_CH, :] = 0.8 + 0.0j
                tx_streamer.send(buf, tx_md)
                tx_md.has_time_spec = False
        finally:
            tx_md.end_of_burst = True
            tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), tx_md)

    tx_thr = threading.Thread(target=_tx_loop, name="TX_loopback_thread")
    tx_thr.start()

    tx_meta_thr = tx_meta_thread(tx_streamer, quit_event)

    rx_thr = rx_thread(
        usrp, rx_streamer, quit_event,
        duration=CAPTURE_TIME, res=result_queue, start_time=start_time
    )

    time.sleep(CAPTURE_TIME + delta(usrp, at_time))
    quit_event.set()
    tx_thr.join()
    rx_thr.join()
    tx_meta_thr.join()

    if user_settings:
        user_settings.poke32(0, SWITCH_RESET_MODE)

    quit_event.clear()


# =============================================================================
#                           BF helper: CSI -> server -> (phi_u1, phi_u2)
# =============================================================================
def get_BF_phase_only(ampl_P1, phi_P1, ampl_P2, phi_P2, SERVER_IP, PILOT_PORT):
    """
    发送两用户 CSI（P1/P2），期望 server 返回：
        {"phi_BF_u1": <rad>, "phi_BF_u2": <rad>}
    """
    dealer_socket = context.socket(zmq.DEALER)
    dealer_socket.setsockopt_string(zmq.IDENTITY, HOSTNAME)
    dealer_socket.connect(f"tcp://{SERVER_IP}:{PILOT_PORT}")

    msg = {
        "host": HOSTNAME,
        "ampl_P1": float(ampl_P1),
        "phi_P1": float(phi_P1),
        "ampl_P2": float(ampl_P2),
        "phi_P2": float(phi_P2),
    }
    dealer_socket.send(json.dumps(msg).encode())

    poller = zmq.Poller()
    poller.register(dealer_socket, zmq.POLLIN)
    socks = dict(poller.poll(30000))

    if dealer_socket in socks and socks[dealer_socket] == zmq.POLLIN:
        reply = dealer_socket.recv()
        response = json.loads(reply.decode())
        logger.info("[%s] Received: %s", HOSTNAME, response)

        phi_u1 = float(response["phi_BF_u1"])
        phi_u2 = float(response["phi_BF_u2"])

        dealer_socket.close()
        return phi_u1, phi_u2

    dealer_socket.close()
    raise TimeoutError(f"[{HOSTNAME}] No reply from server, timed out.")


# =============================================================================
#                           CLI + Main
# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="2-user phase-only MU-TX client")
    parser.add_argument("-i", "--ip", type=str, required=False, help="Server IP address")
    parser.add_argument("--tx-u1", type=str, default="tx_u1.bin", help="UE1 tx waveform file (complex64)")
    parser.add_argument("--tx-u2", type=str, default="tx_u2.bin", help="UE2 tx waveform file (complex64)")
    return parser.parse_args()


def main():
    global meas_id
    args = parse_arguments()

    # Load calibration + experiment settings
    try:
        with open(os.path.join(os.path.dirname(__file__), "cal-settings.yml"), "r") as file:
            vars_ = yaml.safe_load(file)
            globals().update(vars_)  # expects SERVER_IP, PILOT_PORT, START_*, FREE_TX_GAIN, REF_RX_GAIN, ...
    except FileNotFoundError:
        logger.error("Calibration file 'cal-settings.yml' not found.")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load cal-settings.yml: %s", e)
        sys.exit(1)

    # Override IP if provided
    if args.ip:
        globals()["SERVER_IP"] = args.ip

    # Resolve waveform paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    tx_u1_path = args.tx_u1 if os.path.isabs(args.tx_u1) else os.path.join(script_dir, args.tx_u1)
    tx_u2_path = args.tx_u2 if os.path.isabs(args.tx_u2) else os.path.join(script_dir, args.tx_u2)

    quit_event = threading.Event()
    result_queue = queue.Queue()

    try:
        fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")
        usrp = uhd.usrp.MultiUSRP("enable_user_regs, " f"fpga={fpga_path}, " "mode_n=integer")
        logger.info("Using Device: %s", usrp.get_pp_string())

        tx_streamer, rx_streamer = setup(usrp, SERVER_IP)

        # Pilot 1 (UE1)
        measure_pilot(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=START_PILOT_1)
        A_P1, phi_RP1 = result_queue.get()
        logger.info("Pilot1 phase: %s rad / %s%s", fmt(phi_RP1), fmt(np.rad2deg(phi_RP1)), DEG)

        # Pilot 2 (UE2)
        measure_pilot(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=START_PILOT_2)
        A_P2, phi_RP2 = result_queue.get()
        logger.info("Pilot2 phase: %s rad / %s%s", fmt(phi_RP2), fmt(np.rad2deg(phi_RP2)), DEG)

        # Loopback
        measure_loopback(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=START_LB)
        _, phi_RL = result_queue.get()
        logger.info("Loopback phase: %s rad / %s%s", fmt(phi_RL), fmt(np.rad2deg(phi_RL)), DEG)

        # Cable phase
        phi_cable = 0.0
        with open(os.path.join(script_dir, "ref-RF-cable.yml"), "r") as phases_yaml:
            phases_dict = yaml.safe_load(phases_yaml) or {}
            if HOSTNAME in phases_dict:
                phi_cable = float(phases_dict[HOSTNAME])
                logger.debug("Applying cable phase correction (deg): %s", fmt(phi_cable))
            else:
                logger.warning("HOSTNAME not found in ref-RF-cable.yml; phi_cable=0")

        # IMPORTANT: 这里保留你原来的“客户端先做负号+电缆补偿”的做法
        # 如果你后续决定让 server 统一处理共轭/符号，记得把这里的负号逻辑改掉。
        phi_P1_send = -phi_RP1 + np.deg2rad(phi_cable)
        phi_P2_send = -phi_RP2 + np.deg2rad(phi_cable)

        # Get BF phases (two users)
        phi_u1, phi_u2 = get_BF_phase_only(A_P1, phi_P1_send, A_P2, phi_P2_send, SERVER_IP, PILOT_PORT)
        logger.info("BF phases: phi_u1=%s rad, phi_u2=%s rad", fmt(phi_u1), fmt(phi_u2))

        # Tell server we're in TX mode (for scope measurement)
        alive_socket = context.socket(zmq.REQ)
        alive_socket.connect(f"tcp://{SERVER_IP}:{5558}")
        alive_socket.send_string(f"{HOSTNAME} TX")
        alive_socket.close()

        # Common calibration phase applied to BOTH users
        base_phase_common = float(phi_RL - np.deg2rad(phi_cable))
        logger.info(
            "Common phase: %s rad / %s%s",
            fmt(base_phase_common),
            fmt(np.rad2deg(base_phase_common)),
            DEG,
        )

        # Start TX
        usrp.set_tx_gain(FREE_TX_GAIN, LOOPBACK_TX_CH)

        start_time = uhd.types.TimeSpec(START_TX)
        logger.info("Starting MU-TX at START_TX=%s", fmt(START_TX))

        quit_event.clear()
        tx_thr = threading.Thread(
            target=tx_ref_mixed,
            name="TX_MU_thread",
            args=(
                usrp,
                tx_streamer,
                quit_event,
                start_time,
                base_phase_common,
                phi_u1,
                phi_u2,
                tx_u1_path,
                tx_u2_path,
                LOOPBACK_TX_CH,   # active TX ch
                0.8,              # active amplitude
            ),
        )
        tx_thr.start()

        tx_meta_thr = tx_meta_thread(tx_streamer, quit_event)
        send_usrp_in_tx_mode(SERVER_IP)

        # run for TX_TIME from cal-settings.yml
        time.sleep(TX_TIME + (START_TX - usrp.get_time_now().get_real_secs()))
        quit_event.set()

        tx_thr.join()
        tx_meta_thr.join()

        logger.info("DONE")

    except Exception as e:
        logger.error("Exception: %s", e)
        quit_event.set()
        time.sleep(1)
        sys.exit(1)

    finally:
        time.sleep(1)
        sys.exit(0)


if __name__ == "__main__":
    while True:
        main()
