from utils.client_com import Client
import signal
import time
import sys
import argparse, shlex
from datetime import datetime, timezone, timedelta
import uhd
import numpy as np
import yaml
import logging
import os

# =============================================================================
#                           Custom Log Formatter
# =============================================================================
# This formatter adds timestamps with fractional seconds to log messages,
# allowing for more precise event timing (useful in measurement systems).
# =============================================================================

class LogFormatter(logging.Formatter):
    """Custom log formatter that prints timestamps with fractional seconds."""

    @staticmethod
    def pp_now():
        """Return the current time of day as a formatted string with milliseconds."""
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)

    def formatTime(self, record, datefmt=None):
        """Override the default time formatter to include fractional seconds."""
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date

# =============================================================================
#                           Logger and Channel Configuration
# =============================================================================
# This section initializes the global logger and defines the
# channel mapping used for reference and loopback measurements.
# =============================================================================

global logger
global begin_time

connected_to_server = False
begin_time = 2.0

# -------------------------------------------------------------------------
# Logger setup
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Stream logs to console
console = logging.StreamHandler()
logger.addHandler(console)

# Custom log format (includes time, level, and thread name)
formatter = LogFormatter(
    fmt="[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s"
)
console.setFormatter(formatter)


"""Parse the command line arguments"""
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str)

args = parser.parse_args()

# Read experiment settings
with open(args.config_file, "r") as f:
    experiment_settings = yaml.safe_load(f)

try:
    frequency = float(experiment_settings.get("frequency", 920e6))
    channel = int(experiment_settings.get("channel", 0))
    gain = float(experiment_settings.get("gain", 80))
    rate = float(experiment_settings.get("rate", 250e3))
    duration = int(experiment_settings.get("duration", 10))
except ValueError as e:
    print("Could not read all settings:", e)
    sys.exit(-1)

client = None 
got_sync = False


def handle_tx_start(command, args):
    print("Received SYNC command:", command, args)
    
    global got_sync
    
    got_sync = True
    

def handle_signal(signum, frame):
    print("\nStopping client...")
    client.stop()


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking

def setup_usrp_clock(usrp, clock_src, num_mboards):
    usrp.set_clock_source(clock_src)

    end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)

    print("Now confirming lock on clock signals...")

    # Lock onto clock signals for all mboards
    for i in range(num_mboards):
        is_locked = usrp.get_mboard_sensor("ref_locked", i)
        while (not is_locked) and (datetime.now() < end_time):
            time.sleep(1e-3)
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
        if not is_locked:
            print("Unable to confirm clock signal locked on board %d", i)
            return False
        else:
            print("Clock signals are locked")
    return True


def setup_usrp_pps(usrp, pps):
    """Setup the PPS source"""
    usrp.set_time_source(pps)
    return True


def setup_usrp(usrp, server_ip, connect=True):
    rate = RATE
    mcr = 20e6
    assert (
        mcr / rate
    ).is_integer(), f"The masterclock rate {mcr} should be an integer multiple of the sampling rate {rate}"
    # Manual selection of master clock rate may also be required to synchronize multiple B200 units in time.
    usrp.set_master_clock_rate(mcr)
    channels = [0, 1]
    setup_usrp_clock(usrp, "external", usrp.get_num_mboards())
    setup_usrp_pps(usrp, "external")
    # smallest as possible (https://files.ettus.com/manual/page_usrp_b200.html#b200_fe_bw)
    rx_bw = 200e3
    for chan in channels:
        usrp.set_rx_rate(rate, chan)
        usrp.set_tx_rate(rate, chan)
        # NOTE DC offset is enabled
        usrp.set_rx_dc_offset(True, chan)
        usrp.set_rx_bandwidth(rx_bw, chan)
        usrp.set_rx_agc(False, chan)
    # specific settings from loopback/REF PLL
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

    usrp.set_rx_gain(LOOPBACK_RX_GAIN, LOOPBACK_RX_CH)
    usrp.set_rx_gain(REF_RX_GAIN, REF_RX_CH)
    # streaming arguments
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels
    # streamers
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)
    # Step1: wait for the last pps time to transition to catch the edge
    # Step2: set the time at the next pps (synchronous for all boards)
    # this is better than set_time_next_pps as we wait till the next PPS to transition and after that we set the time.
    # this ensures that the FPGA has enough time to clock in the new timespec (otherwise it could be too close to a PPS edge)
    logger.info("Waiting for server sync")
    while not got_sync:
        pass
    
    logger.info("Setting device timestamp to 0...")
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    logger.debug("[SYNC] Resetting time.")
    logger.info(f"RX GAIN PROFILE CH0: {usrp.get_rx_gain_names(0)}")
    logger.info(f"RX GAIN PROFILE CH1: {usrp.get_rx_gain_names(1)}")
    # we wait 2 seconds to ensure a PPS rising edge occurs and latches the 0.000s value to both USRPs.
    time.sleep(2)
    tune_usrp(usrp, FREQ, channels, at_time=begin_time)
    logger.info(
        f"USRP has been tuned and setup. ({usrp.get_time_now().get_real_secs()})"
    )
    return tx_streamer, rx_streamer


def config_streamer(channels, usrp):
    st_args = uhd.usrp.StreamArgs("fc32", "fc32")
    st_args.channels = channels
    return usrp.get_tx_stream(st_args)


def tx(duration, tx_streamer, rate, channels):
    print("TX START")
    metadata = uhd.types.TXMetadata()

    buffer_samps = tx_streamer.get_max_num_samps()
    samps_to_send = int(rate*duration)

    signal = np.ones((len(channels), buffer_samps), dtype=np.complex64)
    signal *= np.exp(1j*np.random.rand(len(channels), 1)*2*np.pi)*0.8 # 0.8 to not exceed to 1.0 threshold

    print(signal[:,0])

    send_samps = 0

    while send_samps < samps_to_send:
        samples = tx_streamer.send(signal, metadata)
        send_samps += samples
    # Send EOB to terminate Tx
    metadata.end_of_burst = True
    tx_streamer.send(np.zeros((len(channels), 1), dtype=np.complex64), metadata)
    print("TX END")
    # Help the garbage collection
    return send_samps


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Attempt to open and load calibration settings from the YAML file
        with open(os.path.join(script_dir, "cal-settings.yml"), "r") as file:
            vars = yaml.safe_load(file)
            globals().update(vars)  # update the global variables with the vars in yaml
    except FileNotFoundError:
        logger.error("Calibration file 'cal-settings.yml' not found in the current directory.")
        exit()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing 'cal-settings.yml': {e}")
        exit()
    except Exception as e:
        logger.error(f"Unexpected error while loading calibration settings: {e}")
        exit()
    
    logger.debug(vars)
    
    try:
        # FPGA file path
        fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")

        # Initialize USRP device with custom FPGA image and integer mode
        usrp = uhd.usrp.MultiUSRP(
            "enable_user_regs, " \
            f"fpga={fpga_path}, " \
            "mode_n=integer"
        )
        logger.info("Using Device: %s", usrp.get_pp_string())
   
        client = Client(args.config_file)
        client.on("SYNC", handle_sync)
        client.start()
        logger.debug("Client running...")
   
        # -------------------------------------------------------------------------
        # STEP 0: Preparations
        # -------------------------------------------------------------------------

        # Set up TX and RX streamers and establish connection
        tx_streamer, rx_streamer = setup_usrp(usrp, server_ip, connect=True)

        # Event used to control thread termination
        quit_event = threading.Event()

        margin = 5.0                     # Safety margin for timing
        cmd_time = CAPTURE_TIME + margin # Duration for one measurement step
        start_next_cmd = cmd_time        # Timestamp for the next scheduled command

        # Queue to collect measurement results and communicate between threads
        result_queue = queue.Queue()
    
    except Exception as e:
        logger.error(e)
    
    quit()
   


    client = Client(args.config_file)
    client.on("tx-start", handle_tx_start)
    client.start()
    print("Client running...")
    
    try:
        while client.running:
            if got_start:
                got_start = False
                tx(duration, tx_streamer, rate, [channel])
                client.send("tx-done")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    client.stop()
    client.join()
    print("Client terminated.")
