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

# ---- 1-bit DAC + dithering defaults (can be overridden by cal-settings.yml) ---
DITHER_SIGMA2 = 1.0e-3   # CN(0, sigma2 I) per sample
TX_SCALE = 0.9           # extra digital scaling to avoid saturation/clipping
# -----------------------------------------------------------------------------

results = []

SWITCH_LOOPBACK_MODE = 0x00000006
SWITCH_RESET_MODE = 0x00000000

context = zmq.Context()

iq_socket = context.socket(zmq.PUB)
iq_socket.bind(f"tcp://*:{50001}")

HOSTNAME = socket.gethostname()[4:]
file_open = False


# =============================================================================
#                           Custom Log Formatter
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
    REF_RX_CH = FREE_TX_CH = 0
    LOOPBACK_RX_CH = LOOPBACK_TX_CH = 1
    logger.debug("\nPLL REF → CH0 RX\nCH1 TX → CH1 RX\nCH0 TX →")
else:
    LOOPBACK_RX_CH = FREE_TX_CH = 0
    REF_RX_CH = LOOPBACK_TX_CH = 1
    logger.debug("\nPLL REF → CH1 RX\nCH1 TX → CH0 RX\nCH0 TX →")


# =============================================================================
#                    1-bit DAC Quantization + Dithering
# =============================================================================
def quantize_1bit(x: np.ndarray, eta: float) -> np.ndarray:
    """
    1-bit complex quantizer:
        Q(z) = sqrt(eta/2) * (sgn(Re{z}) + j*sgn(Im{z}))
    """
    re = np.where(np.real(x) >= 0, 1.0, -1.0)
    im = np.where(np.imag(x) >= 0, 1.0, -1.0)
    return (np.sqrt(eta / 2.0) * (re + 1j * im)).astype(np.complex64)


def gaussian_dither(shape, sigma2: float) -> np.ndarray:
    """
    Complex circular Gaussian dither CN(0, sigma2 I)
    """
    sigma = np.sqrt(float(sigma2))
    return ((sigma / np.sqrt(2.0)) * (np.random.randn(*shape) + 1j * np.random.randn(*shape))).astype(
        np.complex64
    )


# =============================================================================
# RX
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
                            logger.error(
                                "more samples received than buffer long, not storing the data"
                            )
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
        rx_streamer.issue_stream_cmd(
            uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        )

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

        avg_ampl = np.mean(np.abs(iq_samples), axis=1)
        A_rms = np.sqrt(np.mean(np.abs(iq_samples) ** 2, axis=1))

        # Keep your existing convention: put CH1 RMS and phase diff
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

        logger.debug(
            "AVG AMPL IQ CH0: %s CH1: %s",
            fmt(avg_ampl[0]),
            fmt(avg_ampl[1]),
        )


# =============================================================================
# Setup helpers
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
    meas_id, unique_id = sync_socket.recv_string().split(" ")

    file_name = f"data_{HOSTNAME}_{unique_id}_{meas_id}"

    if not file_open:
        data_file = open(f"data_{HOSTNAME}_{unique_id}.txt", "a")
        file_open = True

    logger.debug(meas_id)

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

    # TX gains
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

    # RX gains from YAML (these must exist in cal-settings.yml)
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


# =============================================================================
# Thread helpers
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
            else:
                if async_metadata.event_code != uhd.types.TXMetadataEventCode.burst_ack:
                    logger.error(async_metadata.event_code)
    except KeyboardInterrupt:
        pass


def delta(usrp, at_time):
    return at_time - usrp.get_time_now().get_real_secs()


def tx_thread(usrp, tx_streamer, quit_event, phase=[0, 0], amplitude=[0.8, 0.8], start_time=None):
    tx_thr = threading.Thread(
        target=tx_ref,
        args=(usrp, tx_streamer, quit_event, phase, amplitude, start_time),
    )
    tx_thr.name = "TX_thread"
    tx_thr.start()
    return tx_thr


# =============================================================================
# TX (MODIFIED): Linear precoding + Gaussian dithering + 1-bit quantization
# =============================================================================
def tx_ref(usrp, tx_streamer, quit_event, phase, amplitude, start_time=None):
    """
    MODIFIED for 1-bit DAC verification (per-antenna 1-bit complex quantization + dithering).

    Interpretation:
      - phase[] carries your intended relative phase per TX channel (you pass tx_phase in phase[LOOPBACK_TX_CH])
      - We build a 1-stream precoder vector w = [1, exp(j*phase_ch1)] / sqrt(N)
      - Generate a simple constant symbol s (or you can replace with random QPSK/16QAM later)
      - x = w*s
      - x_q = Q(x + d), Q is 1-bit complex quantizer, d ~ CN(0, sigma^2 I)
      - Send x_q continuously

    Notes:
      - amplitude[] is kept in signature to keep your pipeline unchanged,
        but the 1-bit signal magnitude is controlled by eta + TX_SCALE.
    """
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()

    phase = np.asarray(phase, dtype=np.float64)

    # Paper-style normalization: eta = 1/N
    N = int(num_channels)
    eta = 1.0 / float(N)

    # Precoder vector w (N x 1).
    # We assume ch0 is reference = 1, and ch1 gets the requested phase.
    w = np.zeros((N, 1), dtype=np.complex64)
    w[0, 0] = 1.0 + 0j

    if N > 1:
        w[1, 0] = np.exp(1j * float(phase[1]))
    # If you ever have N>2, extend here:
    for k in range(2, N):
        w[k, 0] = 1.0 + 0j

    w /= np.sqrt(N)

    # Simple constant symbol (unit power)
    s = (1.0 + 1j) / np.sqrt(2.0)

    x = w * np.complex64(s)  # (N, 1)

    # Build a long block
    L = 1000 * int(max_samps_per_packet)
    x_block = np.repeat(x, L, axis=1)  # (N, L)

    # Dither + 1-bit quantization
    d = gaussian_dither(x_block.shape, sigma2=DITHER_SIGMA2)
    x_q = quantize_1bit(x_block + d, eta=eta)

    transmit_buffer = (TX_SCALE * x_q).astype(np.complex64)

    # Timed TX metadata
    tx_md = uhd.types.TXMetadata()
    if start_time is not None:
        tx_md.time_spec = start_time
    else:
        tx_md.time_spec = uhd.types.TimeSpec(
            usrp.get_time_now().get_real_secs() + INIT_DELAY
        )
    tx_md.has_time_spec = True

    try:
        # First send uses the time spec
        tx_streamer.send(transmit_buffer, tx_md)
        tx_md.has_time_spec = False

        while not quit_event.is_set():
            tx_streamer.send(transmit_buffer, tx_md)

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
# Measurement blocks
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

    # NOTE: Loopback uses the same tx_ref() function (now 1-bit). If you want loopback to remain "analog clean"
    # (no 1-bit), you can add a flag and use the old tx_ref for loopback.
    tx_thr = tx_thread(
        usrp,
        tx_streamer,
        quit_event,
        amplitude=amplitudes,
        phase=[0.0, 0.0],
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
# CSI exchange
# =============================================================================
def get_BF(ampl_P1, phi_P1, ampl_P2, phi_P2):
    import json

    dealer_socket = context.socket(zmq.DEALER)
    dealer_socket.setsockopt_string(zmq.IDENTITY, HOSTNAME)
    dealer_socket.connect(f"tcp://{SERVER_IP}:{PILOT_PORT}")

    logger.debug("Sending CSI")

    msg = {
        "host": HOSTNAME,
        "ampl_P1": float(ampl_P1),
        "phi_P1": float(phi_P1),
        "ampl_P2": float(ampl_P2),
        "phi_P2": float(phi_P2),
    }

    dealer_socket.send(json.dumps(msg).encode())
    logger.debug("Message sent, waiting for response...")

    poller = zmq.Poller()
    poller.register(dealer_socket, zmq.POLLIN)
    socks = dict(poller.poll(30000))

    result = None
    if dealer_socket in socks and socks[dealer_socket] == zmq.POLLIN:
        reply = dealer_socket.recv()
        logger.debug("Raw reply: %r", reply)
        response = json.loads(reply.decode())
        logger.info("[%s] Received: %s", HOSTNAME, response)
        result = response["phi_BF"]
        logger.debug("Received response: %s", result)
    else:
        logger.warning("[%s] No reply from server, timed out.", HOSTNAME)

    dealer_socket.close()
    return result


# =============================================================================
# TX with phase correction (uses the new 1-bit tx_ref())
# =============================================================================
def tx_phase_coh(usrp, tx_streamer, quit_event, phase_corr, at_time, long_time=True):
    logger.debug("########### TX with adjusted phases (1-bit + dither) ###########")

    phases = [0.0, 0.0]
    amplitudes = [0.0, 0.0]

    phases[LOOPBACK_TX_CH] = phase_corr
    amplitudes[LOOPBACK_TX_CH] = 0.8

    logger.debug(f"Phases: {phases}")
    logger.debug(f"amplitudes: {amplitudes}")
    logger.debug(f"TX Gain: {FREE_TX_GAIN}")

    usrp.set_tx_gain(FREE_TX_GAIN, LOOPBACK_TX_CH)

    start_time = uhd.types.TimeSpec(at_time)

    tx_thr = tx_thread(
        usrp,
        tx_streamer,
        quit_event,
        amplitude=amplitudes,
        phase=phases,
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

    logger.debug("Transmission completed successfully")

    quit_event.clear()
    return tx_thr, tx_meta_thr


# =============================================================================
# CLI
# =============================================================================
def parse_arguments():
    global SERVER_IP

    parser = argparse.ArgumentParser(description="Beamforming control script (1-bit DAC verification)")
    parser.add_argument("-i", "--ip", type=str, help="IP address of the server (optional)", required=False)
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


# =============================================================================
# main
# =============================================================================
def main():
    global meas_id, file_name_state

    args = parse_arguments()

    # Load calibration/settings (overrides globals, including DITHER_SIGMA2, TX_SCALE)
    try:
        with open(os.path.join(os.path.dirname(__file__), "cal-settings.yml"), "r") as file:
            vars = yaml.safe_load(file)
            globals().update(vars)
    except FileNotFoundError:
        logger.error("Calibration file 'cal-settings.yml' not found in the current directory.")
        exit()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing 'cal-settings.yml': {e}")
        exit()
    except Exception as e:
        logger.error(f"Unexpected error while loading calibration settings: {e}")
        exit()

    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")

        usrp = uhd.usrp.MultiUSRP(
            "enable_user_regs, " f"fpga={fpga_path}, " "mode_n=integer"
        )
        logger.info("Using Device: %s", usrp.get_pp_string())

        # Setup
        tx_streamer, rx_streamer = setup(usrp, SERVER_IP, connect=True)

        quit_event = threading.Event()
        result_queue = queue.Queue()

        # Pilot 1
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

        # Pilot 2
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

        # Loopback
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

        # Cable phase correction
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
                print(exc)

        phi_BF = get_BF(
            A_P1, -phi_RP1 + np.deg2rad(phi_cable),
            A_P2, -phi_RP2 + np.deg2rad(phi_cable),
        )

        if BEAMFORMER == "MRT":
            phi_BF = phi_RP2 - np.deg2rad(phi_cable)

        alive_socket = context.socket(zmq.REQ)
        alive_socket.connect(f"tcp://{SERVER_IP}:{5558}")
        logger.debug("Sending TX MODE")
        alive_socket.send_string(f"{HOSTNAME} TX")
        alive_socket.close()

        # Final phase for coherent TX
        tx_phase = phi_RL - np.deg2rad(phi_cable) + phi_BF
        logger.info(
            "Phase correction: %s (rad) / %s%s",
            fmt(tx_phase),
            fmt(np.rad2deg(tx_phase)),
            DEG,
        )

        # TX (1-bit + dither)
        tx_phase_coh(
            usrp,
            tx_streamer,
            quit_event,
            phase_corr=tx_phase,
            at_time=START_TX,
            long_time=True,
        )

        print("DONE")

    except Exception as e:
        logger.debug("Sending signal to stop!")
        logger.error(e)
        quit_event.set()

    finally:
        time.sleep(1)
        sys.exit(0)


if __name__ == "__main__":
    while 1:
        main()
