import zmq
import json
import time
import signal
import threading
from datetime import datetime, timedelta

class Server:
    def __init__(self, bind="tcp://*:5678", heartbeat_timeout=10, silent=False):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind(bind)
        self.clients = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.silent = silent
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
                    frames = self.sock.recv_multipart()
                    if not frames:
                        continue

                    identity, *payload = frames

                    # First payload frame is the message type
                    msg_type = payload[0].decode() if len(payload) >= 1 else ""
                    msg_payload = payload[1:] if len(payload) > 1 else []

                    # Update last_seen
                    self.clients[identity] = {"last_seen": datetime.utcnow()}

                    # Handle messages
                    if msg_type == "heartbeat":
                        if not self.silent:
                            print(f"[HEARTBEAT] {identity.decode()}")
                    elif msg_type == "response":
                        if not self.silent:
                            print(f"[RESPONSE] {identity.decode()}: {msg_payload}")

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
            if not self.silent:
                print(f"[TIMEOUT] Removing client {cid.decode()}")
            del self.clients[cid]
                
    def print_clients(self, short=False):
        if len(self.clients) == 0:
            print("no connected clients")
        else:
            if short:
                cids = sorted(self.clients)
                cidstr = ""
                for cid in cids:
                    if len(cidstr) > 0:
                        cidstr += " "
                    cidstr += cid.decode();
                print(cidstr)
            else:
                print("connected clients:")
                for cid, info in list(self.clients.items()):
                    print(cid, "- last seen:", info["last_seen"])
