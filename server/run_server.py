from utils.server_com import Server
import signal
import time

server = Server()

def handle_signal(signum, frame):
    print("\nReceived signal, stopping server...")
    server.stop()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

if __name__ == "__main__":
    server.start()   # <-- non-blocking
    print("Server running in background thread.")

    # Main thread idle loop
    try:
        while server.running:
            server.print_clients()
            time.sleep(5)   # do whatever else you want here
    except KeyboardInterrupt:
        pass

    server.stop()
    server.join()
    print("Server terminated.")
