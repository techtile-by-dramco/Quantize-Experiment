#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import socket
import sys
import threading
import time
from datetime import datetime, timedelta
import numpy as np
import uhd
import yaml
import tools
import argparse
import zmq
import queue

# =============================================================================
#                           Experiment Configuration
# =============================================================================
CMD_DELAY = 0.05  # Command delay (50 ms) between USRP instructions
RX_TX_SAME_CHANNEL = True  # True if loopback occurs between the same RF channel
CLOCK_TIMEOUT = 1000  # Timeout for external clock locking (in ms)
INIT_DELAY = 0.2  # Initial delay before starting transmission (200 ms)
RATE = 250e3  # Sampling rate in samples per second (250 kSps)
LOOPBACK_TX_GAIN = 50  # Empirically determined transmit gain for loopback tests
RX_GAIN = 22  # Not directly used below (kept for compatibility)
CAPTURE_TIME = 10  # Duration of each capture in seconds
FREQ = 0  # Base frequency offset (Hz); 0 means use default center frequency
meas_id = 0  # Measurement identifier
exp_id = 0  # Experiment identifier

results = []

SWITCH_LOOPBACK_MODE = 0x00000006  # which is 110
SWITCH_RESET_MODE = 0x00000000

context = zmq.Context()
iq_socket = context.socket(zmq.PUB)
iq_socket.bind(f"tcp://*:{50001}")

HOSTNAME = socket.gethostname()[4:]
file_open = False
# === MU-ZF weights from server (per-AP complex) ===
W_U1 = None  # complex
W_U2 = None  # complex
# =============================================================================
#                   1-bit DAC Quantization (with optional dithering)
# =============================================================================
ENABLE_1BIT_DAC = False          # True: enable 1-bit quantizer after precoding
ENABLE_DITHER = False            # True: add dither before 1-bit quantization
DITHER_REL_STD = 0.10           # dither std relative to signal RMS amplitude (per sample)
ONEBIT_POWER_NORM = True        # normalize 1-bit output to match input RMS power
DITHER_SEED_BASE = 12345        # reproducible base seed
# =============================================================================


# =============================================================================
#                           Custom Log Formatter
# =============================================================================
class LogFormatter(logging.Formatter):
    """Custom log formatter that prints timestamps with fractional seconds."""

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
    """Console formatter with ANSI colors per level."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def fmt(val):
    """Format float to 0.3f."""
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


DEG = "\u00b0"


# =============================================================================
#                           Logger and Channel Configuration
# =============================================================================
global logger
global begin_time

connected_to_server = False
begin_time = 2.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
logger.addHandler(console)

formatter = LogFormatter(
    fmt="[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s"
)
console.setFormatter(ColoredFormatter(fmt=formatter._fmt))

file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "log.txt"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

TOPIC_CH0 = b"CH0"
TOPIC_CH1 = b"CH1"

if RX_TX_SAME_CHANNEL:
    # Reference signal received on CH0, loopback on CH1
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
    """
    Add zero-mean complex Gaussian dither to x.
    rel_std is relative to RMS(|x|). Dither is applied per sample.
    """
    if rel_std <= 0:
        return x
    if x.size == 0:
        return x

    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if rms <= 0:
        return x

    sigma = rel_std * rms
    d = (rng.normal(0.0, sigma / np.sqrt(2), size=x.shape) +
         1j * rng.normal(0.0, sigma / np.sqrt(2), size=x.shape)).astype(np.complex64)
    return x + d


def one_bit_quantize_complex(x: np.ndarray, power_norm: bool = True, eps: float = 1e-12) -> np.ndarray:
    """
    1-bit quantization per real/imag part:
        q = sign(Re{x}) + j*sign(Im{x})
    Optional: normalize q to match the RMS power of x.
    """
    if x.size == 0:
        return x.astype(np.complex64)

    re = np.real(x)
    im = np.imag(x)

    # sign(0)->+1 (deterministic)
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
    """
    Load tx.bin as complex64 ONLY.
    Assumes file is stored as np.complex64 raw samples (8 bytes/sample).
    """
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
            try:
                num_rx_i = rx_streamer.recv(recv_buffer, rx_md, timeout)
                if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
                    logger.error(rx_md.error_code)
                else:
                    if num_rx_i > 0:
                        samples = recv_buffer[:, :num_rx_i]
                        if num_rx + num_rx_i > buffer_length:
                            logger.error("more samples received than buffer long, not storing the data")
                        else:
                            iq_data[:, num_rx: num_rx + num_rx_i] = samples
                            num_rx += num_rx_i
            except RuntimeError as ex:
                logger.error("Runtime error in receive: %s", ex)
                return
    except KeyboardInterrupt:
        pass
    finally:
        logger.debug("CTRL+C is pressed or duration is reached, closing off ")
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

        # Drop first 1s to avoid transients
        iq_samples = iq_data[:, int(RATE * 1): num_rx]

        phase_ch0, freq_slope_ch0_before, freq_slope_ch0_after = tools.get_phases_and_apply_bandpass(
            iq_samples[0, :]
        )
        phase_ch1, freq_slope_ch1_before, freq_slope_ch1_after = tools.get_phases_and_apply_bandpass(
            iq_samples[1, :]
        )

        logger.debug(
            "Frequency offset CH0:     %.2f Hz     %.2f Hz",
            float(freq_slope_ch0_before),
            float(freq_slope_ch0_after),
        )
        logger.debug(
            "Frequency offset CH1:     %.2f Hz     %.2f Hz",
            float(freq_slope_ch1_before),
            float(freq_slope_ch1_after),
        )

        phase_diff = tools.to_min_pi_plus_pi(phase_ch0 - phase_ch1, deg=False)

        logger.debug(
            "Phase CH1: mean %s%s min %s%s max %s%s",
            fmt(np.rad2deg(tools.circmean(phase_diff, deg=False))),
            DEG,
            fmt(np.rad2deg(phase_diff).min()),
            DEG,
            fmt(np.rad2deg(phase_diff).max()),
            DEG,
        )

        _circ_mean = tools.circmean(phase_diff, deg=False)

        A_rms = np.sqrt(np.mean(np.abs(iq_samples) ** 2, axis=1))
        result_queue.put((A_rms[1], _circ_mean))

        max_I = np.max(np.abs(np.real(iq_samples)), axis=1)
        max_Q = np.max(np.abs(np.imag(iq_samples)), axis=1)

        logger.debug(
            "MAX AMPL IQ CH0: I %s Q %s CH1: I %s Q %s",
            fmt(max_I[0]),
            fmt(max_Q[0]),
            fmt(max_I[1]),
            fmt(max_Q[1]),
        )

        avg_ampl = np.mean(np.abs(iq_samples), axis=1)
        logger.debug("AVG AMPL IQ CH0: %s CH1: %s", fmt(avg_ampl[0]), fmt(avg_ampl[1]))


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
        else:
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
        print(".")
        time.sleep(0.01)
    logger.info("RX LO is locked")

    while not usrp.get_tx_sensor("lo_locked").to_bool():
        print(".")
        time.sleep(0.01)
    logger.info("TX LO is locked")


# =============================================================================
#                           Server Sync
# =============================================================================
def wait_till_go_from_server(ip, _connect=True):
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


def setup(usrp, SERVER_IP, connect=True):
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

    # These are overridden by cal-settings.yml below (FREE_TX_GAIN/REF_RX_GAIN/...)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

    usrp.set_rx_gain(LOOPBACK_RX_GAIN, LOOPBACK_RX_CH)
    usrp.set_rx_gain(REF_RX_GAIN, REF_RX_CH)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels

    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)

    wait_till_go_from_server(SERVER_IP, connect)

    logger.info("Setting device timestamp to 0...")
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    logger.debug("[SYNC] Resetting time.")
    logger.info(f"RX GAIN PROFILE CH0: {usrp.get_rx_gain_names(0)}")
    logger.info(f"RX GAIN PROFILE CH1: {usrp.get_rx_gain_names(1)}")

    time.sleep(2)
    tune_usrp(usrp, FREQ, channels, at_time=begin_time)

    logger.info(f"USRP has been tuned and setup. ({usrp.get_time_now().get_real_secs()})")
    return tx_streamer, rx_streamer

def rx_thread(usrp, rx_streamer, quit_event, duration, res, start_time=None):
    _rx_thread = threading.Thread(
        target=rx_ref,
        args=(usrp, rx_streamer, quit_event, duration, res, start_time),
    )
    _rx_thread.name = "RX_thread"
    _rx_thread.start()
    return _rx_thread

# =============================================================================
#                           Thread wrappers
# =============================================================================
def tx_thread(usrp, tx_streamer, quit_event, w_u1, w_u2, phase_hw, start_time=None):
    tx_thr = threading.Thread(
        target=tx_ref,
        args=(usrp, tx_streamer, quit_event, w_u1, w_u2, phase_hw, start_time),
    )
    tx_thr.name = "TX_thread"
    tx_thr.start()
    return tx_thr



def tx_async_th(tx_streamer, quit_event):
    async_metadata = uhd.types.TXAsyncMetadata()
    try:
        while not quit_event.is_set():
            if not tx_streamer.recv_async_msg(async_metadata, 0.01):
                continue
            else:
                if async_metadata.event_code != uhd.types.TXMetadataEventCode.burst_ack:
                    logger.error(async_metadata.event_code)
    except KeyboardInterrupt:
        pass


def delta(usrp, at_time):
    return at_time - usrp.get_time_now().get_real_secs()


def get_current_time(usrp):
    return usrp.get_time_now().get_real_secs()

def tx_thread_loopback(usrp, tx_streamer, quit_event, phase=[0.0, 0.0], amplitude=[0.0, 0.0], start_time=None):
    """
    Loopback-only TX thread: sends a constant complex tone per channel:
        sample[ch] = amplitude[ch] * exp(1j*phase[ch])
    phase MUST be in radians.
    """
    tx_thr = threading.Thread(
        target=tx_ref_tone,
        args=(usrp, tx_streamer, quit_event, phase, amplitude, start_time),
    )
    tx_thr.name = "TX_LB_thread"
    tx_thr.start()
    return tx_thr


def tx_ref_tone(usrp, tx_streamer, quit_event, phase, amplitude, start_time=None):
    """
    Loopback-only transmitter: constant complex baseband per channel.
    This matches your single-user single-tone approach (no tx1/tx2 involved).
    phase is in radians.
    """
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()

    amplitude = np.asarray(amplitude, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)

    # Constant complex value per channel
    sample = (amplitude * np.exp(1j * phase)).astype(np.complex64)

    # Big buffer with constant samples
    transmit_buffer = np.ones((num_channels, 1000 * max_samps_per_packet), dtype=np.complex64)
    for ch in range(num_channels):
        transmit_buffer[ch, :] *= sample[ch]

    tx_md = uhd.types.TXMetadata()
    if start_time is not None:
        tx_md.time_spec = start_time
    else:
        tx_md.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)

    tx_md.has_time_spec = True
    tx_md.end_of_burst = False

    try:
        while not quit_event.is_set():
            tx_streamer.send(transmit_buffer, tx_md)
            tx_md.has_time_spec = False  # only first burst timed
    except KeyboardInterrupt:
        logger.debug("CTRL+C detected — stopping loopback tone TX")
    finally:
        tx_md.end_of_burst = True
        tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), tx_md)


# =============================================================================
#                           TX: tx.bin waveform repeat (complex64)
# =============================================================================
def tx_ref(usrp, tx_streamer, quit_event, w_u1, w_u2, phase_hw, start_time=None):
    """
    True MU transmit (single-carrier, code-division):
        x(t) = w_u1 * s1(t) + w_u2 * s2(t)
    where s1=tx1.bin, s2=tx2.bin (complex64).
    """
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()

    # We only transmit on LOOPBACK_TX_CH as before (distributed AP single TX chain)
    # Keep other channels zero.
    tx_ch = LOOPBACK_TX_CH

    script_dir = os.path.dirname(os.path.realpath(__file__))
    tx1_path = os.path.join(script_dir, "tx1.bin")
    tx2_path = os.path.join(script_dir, "tx2.bin")

    s1 = load_tx_bin_complex64(tx1_path)  # 1-D complex64
    s2 = load_tx_bin_complex64(tx2_path)  # 1-D complex64
    if s1.size != s2.size:
        raise ValueError(f"tx1.bin and tx2.bin must have same length: {s1.size} vs {s2.size}")

    L = int(s1.size)
    logger.info("Loaded MU sequences: tx1.bin=%s tx2.bin=%s (L=%d)", tx1_path, tx2_path, L)

    # Optional: normalize s1,s2 to unit RMS (safe even if already unit power)
    s1 = (s1 / (np.sqrt(np.mean(np.abs(s1)**2)) + 1e-12)).astype(np.complex64, copy=False)
    s2 = (s2 / (np.sqrt(np.mean(np.abs(s2)**2)) + 1e-12)).astype(np.complex64, copy=False)

    # Reproducible RNG per measurement id
    try:
        seed = int(DITHER_SEED_BASE + int(meas_id))
    except Exception:
        seed = int(DITHER_SEED_BASE)
    rng = np.random.default_rng(seed)

    tx_md = uhd.types.TXMetadata()
    if start_time is not None:
        tx_md.time_spec = start_time
    else:
        tx_md.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)
    tx_md.has_time_spec = True
    tx_md.end_of_burst = False

    idx = 0
    try:
        while not quit_event.is_set():
            n = int(max_samps_per_packet)
            if n <= 0:
                continue

            # slice with wrap-around
            if idx + n <= L:
                c1 = s1[idx:idx + n]
                c2 = s2[idx:idx + n]
                idx += n
                if idx >= L:
                    idx = 0
            else:
                first_len = L - idx
                c1 = np.concatenate([s1[idx:], s1[:(n - first_len)]])
                c2 = np.concatenate([s2[idx:], s2[:(n - first_len)]])
                idx = n - first_len

            transmit_buffer = np.zeros((num_channels, n), dtype=np.complex64)

            # ---- MU superposition on tx_ch ----
            hw_rot = np.exp(1j * np.float32(phase_hw)).astype(np.complex64)
            x = (hw_rot * (w_u1 * c1 + w_u2 * c2)).astype(np.complex64, copy=False)

            if ENABLE_DITHER:
                x = add_complex_dither(x, rel_std=DITHER_REL_STD, rng=rng)

            if ENABLE_1BIT_DAC:
                x = one_bit_quantize_complex(x, power_norm=ONEBIT_POWER_NORM)

            transmit_buffer[tx_ch, :] = x

            tx_streamer.send(transmit_buffer, tx_md)
            tx_md.has_time_spec = False  # only first packet timed

    except KeyboardInterrupt:
        logger.debug("CTRL+C detected — stopping transmission")
    finally:
        tx_md.end_of_burst = True
        tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), tx_md)



def tx_meta_thread(tx_streamer, quit_event):
    tx_meta_thr = threading.Thread(target=tx_async_th, args=(tx_streamer, quit_event))
    tx_meta_thr.name = "TX_META_thread"
    tx_meta_thr.start()
    return tx_meta_thr


def starting_in(usrp, at_time):
    return f"Starting in {delta(usrp, at_time):.2f}s"


# =============================================================================
#                           Measurements
# =============================================================================
def measure_pilot(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=None):
    logger.debug("########### Measure PILOT ###########")
    usrp.set_rx_antenna("TX/RX", 1)

    start_time = uhd.types.TimeSpec(at_time)
    logger.debug(starting_in(usrp, at_time))

    rx_thr = rx_thread(
        usrp,
        rx_streamer,
        quit_event,
        duration=CAPTURE_TIME,
        res=result_queue,
        start_time=start_time,
    )

    time.sleep(CAPTURE_TIME + delta(usrp, at_time))

    quit_event.set()
    rx_thr.join()

    usrp.set_rx_antenna("RX2", 1)
    quit_event.clear()


def measure_loopback(usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=None):
    logger.debug("########### Measure LOOPBACK ###########")

    amplitudes = [0.0, 0.0]
    amplitudes[LOOPBACK_TX_CH] = 0.8

    start_time = uhd.types.TimeSpec(at_time)
    logger.debug(starting_in(usrp, at_time))

    user_settings = None
    try:
        user_settings = usrp.get_user_settings_iface(1)
        if user_settings:
            logger.debug(user_settings.peek32(0))
            user_settings.poke32(0, SWITCH_LOOPBACK_MODE)
            logger.debug(user_settings.peek32(0))
        else:
            logger.error("Cannot write to user settings.")
    except Exception as e:
        logger.error(e)

    tx_thr = tx_thread_loopback(
        usrp,
        tx_streamer,
        quit_event,
        phase=[0.0, 0.0], 
        amplitude=amplitudes,
        start_time=start_time,
    )

    tx_meta_thr = tx_meta_thread(tx_streamer, quit_event)

    rx_thr = rx_thread(
        usrp,
        rx_streamer,
        quit_event,
        duration=CAPTURE_TIME,
        res=result_queue,
        start_time=start_time,
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
#                           BF helper (CSI -> server -> phi_BF)
# =============================================================================
def get_BF(ampl_P1, phi_P1, ampl_P2, phi_P2):
    """
    Send CSI (2 users) to server and receive MU-ZF weights (w_u1, w_u2) for THIS AP.
    We also apply per-AP power normalization to avoid clipping:
        |w_u1|^2 + |w_u2|^2 <= 1
    """
    import json

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

    if not (dealer_socket in socks and socks[dealer_socket] == zmq.POLLIN):
        dealer_socket.close()
        raise TimeoutError(f"[{HOSTNAME}] No reply from server (pilot weights timed out).")

    reply = dealer_socket.recv()
    response = json.loads(reply.decode())
    dealer_socket.close()

    w_u1 = complex(float(response["w_u1_re"]), float(response["w_u1_im"]))
    w_u2 = complex(float(response["w_u2_re"]), float(response["w_u2_im"]))

    # ---- per-AP power normalization (important!) ----
    # This ensures the combined transmit power (before global TX gain) is bounded.
    p = (abs(w_u1) ** 2 + abs(w_u2) ** 2)
    if p > 1e-12:
        scale = 1.0 / np.sqrt(max(p, 1e-12))
        w_u1 *= scale
        w_u2 *= scale

    logger.info("[%s] MU-ZF weights: w_u1=%s w_u2=%s |w|^2=%.3f",
                HOSTNAME, w_u1, w_u2, (abs(w_u1)**2 + abs(w_u2)**2))

    return w_u1, w_u2



def tx_phase_coh(usrp, tx_streamer, quit_event, w_u1: complex, w_u2: complex, phase_hw: float, at_time, long_time=True):
    """
    MU-ZF TX with hardware/common phase compensation:
        x(t) = exp(j*phase_hw) * (w_u1*s1(t) + w_u2*s2(t))
    phase_hw accounts for loopback + cable correction.
    """
    logger.debug("########### MU-ZF TX (tx1+tx2) + HW phase corr ###########")

    usrp.set_tx_gain(FREE_TX_GAIN, LOOPBACK_TX_CH)

    start_time = uhd.types.TimeSpec(at_time)

    # IMPORTANT: pass phase_hw down to tx_ref via tx_thread
    tx_thr = tx_thread(
        usrp,
        tx_streamer,
        quit_event,
        w_u1=w_u1,
        w_u2=w_u2,
        phase_hw=phase_hw,
        start_time=start_time,
    )

    tx_meta_thr = tx_meta_thread(tx_streamer, quit_event)

    send_usrp_in_tx_mode(SERVER_IP)

    if long_time:
        time.sleep(TX_TIME + delta(usrp, at_time))
    else:
        time.sleep(10.0 + delta(usrp, at_time))

    quit_event.set()
    tx_thr.join()
    tx_meta_thr.join()

    logger.debug("MU-ZF transmission completed successfully")
    quit_event.clear()
    return tx_thr, tx_meta_thr



# =============================================================================
#                           CLI + Main
# =============================================================================
def parse_arguments():
    global SERVER_IP

    parser = argparse.ArgumentParser(description="Beamforming control script")
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        help="IP address of the server (optional)",
        required=False,
    )
    parser.add_argument("--config-file", type=str)
    parser.add_argument(
        "--tx-phase-file",
        type=str,
        default="tx-phases-smc2-old.yml",
        help="Path to TX phase YAML (default: tx-phases-smc2-old.yml)",
    )

    args = parser.parse_args()

    logger.info("Invocation args: %s", " ".join(sys.argv))

    if args.ip:
        logger.debug(f"Setting server IP to: {args.ip}")
        SERVER_IP = args.ip
    return args


def main():
    global meas_id

    args = parse_arguments()

    try:
        with open(os.path.join(os.path.dirname(__file__), "cal-settings.yml"), "r") as file:
            vars_ = yaml.safe_load(file)
            globals().update(vars_)
    except FileNotFoundError:
        logger.error("Calibration file 'cal-settings.yml' not found in the current directory.")
        exit()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing 'cal-settings.yml': {e}")
        exit()
    except Exception as e:
        logger.error(f"Unexpected error while loading calibration settings: {e}")
        exit()

    quit_event = None
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")

        usrp = uhd.usrp.MultiUSRP(
            "enable_user_regs, " f"fpga={fpga_path}, " "mode_n=integer"
        )
        logger.info("Using Device: %s", usrp.get_pp_string())

        # STEP 0: setup + sync
        tx_streamer, rx_streamer = setup(usrp, SERVER_IP, connect=True)

        quit_event = threading.Event()
        result_queue = queue.Queue()

        # STEP 1: Pilot 1
        measure_pilot(
            usrp,
            tx_streamer,
            rx_streamer,
            quit_event,
            result_queue,
            at_time=START_PILOT_1,
        )
        A_P1, phi_RP1 = result_queue.get()
        logger.info(
            "Phase pilot 1 reference signal: %s (rad) / %s%s",
            fmt(phi_RP1),
            fmt(np.rad2deg(phi_RP1)),
            DEG,
        )

        # STEP 2: Pilot 2
        measure_pilot(
            usrp,
            tx_streamer,
            rx_streamer,
            quit_event,
            result_queue,
            at_time=START_PILOT_2,
        )
        A_P2, phi_RP2 = result_queue.get()
        logger.info(
            "Phase pilot 2 reference signal: %s (rad) / %s%s",
            fmt(phi_RP2),
            fmt(np.rad2deg(phi_RP2)),
            DEG,
        )

        # STEP 3: Loopback
        measure_loopback(
            usrp,
            tx_streamer,
            rx_streamer,
            quit_event,
            result_queue,
            at_time=START_LB,
        )
        _, phi_RL = result_queue.get()
        logger.info(
            "Phase LB reference signal: %s (rad) / %s%s",
            fmt(phi_RL),
            fmt(np.rad2deg(phi_RL)),
            DEG,
        )

        # STEP 4: Cable phase
        phi_cable = 0
        with open(os.path.join(os.path.dirname(__file__), "ref-RF-cable.yml"), "r") as phases_yaml:
            try:
                phases_dict = yaml.safe_load(phases_yaml)
                if HOSTNAME in phases_dict.keys():
                    phi_cable = phases_dict[HOSTNAME]
                    logger.debug(f"Applying cable phase correction: {phi_cable}")
                else:
                    logger.error("Phase offset not found in ref-RF-cable.yml")
            except yaml.YAMLError as exc:
                logger.error(exc)

        # STEP 5: Get MU-ZF weights (two columns) from server for this AP
        w_u1, w_u2 = get_BF(
            A_P1,
            -phi_RP1 + np.deg2rad(phi_cable),
            A_P2,
            -phi_RP2 + np.deg2rad(phi_cable),
        )

        # Inform server TX mode (unchanged)
        alive_socket = context.socket(zmq.REQ)
        alive_socket.connect(f"tcp://{SERVER_IP}:{5558}")
        logger.debug("Sending TX MODE")
        alive_socket.send_string(f"{HOSTNAME} TX")
        alive_socket.close()

        # Hardware/common phase compensation ONLY (keep your original intention)
        phase_hw = float(phi_RL - np.deg2rad(phi_cable))
        logger.info(
            "HW phase compensation: %s (rad) / %s%s",
            fmt(phase_hw),
            fmt(np.rad2deg(phase_hw)),
            DEG,
        )

        # STEP 6: Timed MU-ZF TX: x = exp(j*phase_hw) * (w_u1*tx1 + w_u2*tx2)
        tx_phase_coh(
            usrp,
            tx_streamer,
            quit_event,
            w_u1=w_u1,
            w_u2=w_u2,
            phase_hw=phase_hw,
            at_time=START_TX,
            long_time=True,
        )
        print("DONE")

    except Exception as e:
        logger.debug("Sending signal to stop!")
        logger.error(e)
        if quit_event is not None:
            quit_event.set()

    finally:
        time.sleep(1)
        sys.exit(0)


if __name__ == "__main__":
    while 1:
        main()
