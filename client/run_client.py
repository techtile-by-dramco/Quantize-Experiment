from utils.client_com import Client
import signal
import time
import sys
import argparse

parser = argparse.ArgumentParser(
    description="Run client."
)

parser.add_argument(
    "--config-file", "-c",
    default="",
    help="full path to config file"
)

args = parser.parse_args()

client = Client(args.config_file)

def handle_ping(command, args):
    print("Received PING event:", command, args)

def handle_custom(command, args):
    print("Received sync command:", command, args)
    client.send("ack", "got sync")

def handle_signal(signum, frame):
    print("\nStopping client...")
    client.stop()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

client.on("ping", handle_ping)
client.on("sync", handle_custom)

if __name__ == "__main__":
    client.start()
    print("Client running...")

    try:
        while client.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    client.stop()
    client.join()
    print("Client terminated.")
    sys.exit(0)
