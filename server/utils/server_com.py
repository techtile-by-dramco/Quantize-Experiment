import zmq
import json
import time
import signal
import threading
from datetime import datetime, timedelta

class Server:
    def __init__(self, bind="tcp://*:5555", heartbeat_timeout=10):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind(bind)
        self.clients = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.running = True
        self.thread = None

    def start(self):
        """Start the server in a background thread."""
        if self.thread is not None:
            return  # already running

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        """Ask the server loop to stop."""
        self.running = False

    def join(self):
        """Wait for the server thread to finish."""
        if self.thread is not None:
            self.thread.join()

    def cleanup(self):
        """Close resources cleanly."""
        print("\nShutting down server...")

        try:
            self.sock.close(linger=0)
        except Exception:
            pass

        try:
            self.ctx.term()
        except Exception:
            pass

        print("Server stopped cleanly.")

    def run(self):
        print("Server running... waiting for clients (Ctrl+C or Ctrl+Z to stop)")

        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)

        try:
            while self.running:
                try:
                    socks = dict(poller.poll(1000))  # may be interrupted
                except zmq.error.ZMQError:
                    break
                except KeyboardInterrupt:
                    # KeyboardInterrupt raised during poll()
                    self.running = False
                    break

                if self.sock in socks:
                    try:
                        identity, raw = self.sock.recv_multipart()
                        msg = json.loads(raw.decode())
                    except KeyboardInterrupt:
                        self.running = False
                        break

                    now = datetime.utcnow()
                    self.clients[identity] = {"last_seen": now}

                    if msg["type"] == "register":
                        print(f"[REGISTER] {identity.decode()}")
                    elif msg["type"] == "heartbeat":
                        print(f"[HEARTBEAT] {identity.decode()}")
                    elif msg["type"] == "response":
                        print(f"[RESPONSE] {identity.decode()}: {msg}")

                self.purge_dead()

        except KeyboardInterrupt:
            # Interrupt outside poll, e.g. between iterations
            pass
        finally:
            self.cleanup()

    def purge_dead(self):
        now = datetime.utcnow()
        dead = []
        for cid, info in list(self.clients.items()):
            if now - info["last_seen"] > timedelta(seconds=self.heartbeat_timeout):
                dead.append(cid)
        for cid in dead:
            print(f"[TIMEOUT] Removing client {cid.decode()}")
            del self.clients[cid]
                
    def print_clients(self):
        print("connected clients:")
        for cid, info in list(self.clients.items()):
            print(cid, "- last seen:", info["last_seen"])
