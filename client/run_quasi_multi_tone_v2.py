import signal
import time
import sys
import argparse, shlex
from datetime import datetime, timezone, timedelta
import uhd
import numpy as np
import yaml

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
got_start = False


def handle_tx_start(command, args):
    print("Received tx-start command:", command, args)
    
    global got_start
    global duration
    
    got_start = True
    _, _, val_str = args[0].partition("=")
    duration = int(val_str)
    

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
    """
    create usrp handle
    """    
    usrp_arg_string = f"-a \"type=b200\" -f {frequency} -c {channel} --gain {gain} -d {duration} --noip --rate {rate}" 
    usrp = uhd.usrp.MultiUSRP(usrp_arg_string)
    
    setup_usrp_clock(usrp, "internal", usrp.get_num_mboards())
    setup_usrp_pps(usrp, "external")

    usrp.set_tx_rate(rate, channel)
    usrp.set_tx_freq(uhd.types.TuneRequest(frequency, 0), channel)
    usrp.set_tx_gain(gain, channel)

    while not usrp.get_rx_sensor("lo_locked").to_bool():
        time.sleep(0.01)

    print("RX LO is locked")

    while not usrp.get_tx_sensor("lo_locked").to_bool():
        time.sleep(0.01)

    print("TX LO is locked")
      
    tx_streamer = config_streamer([channel], usrp)

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
