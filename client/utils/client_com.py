import zmq
import threading
import time
import yaml
import socket
import re


class Client:
    def __init__(self, config_path="client_config.yaml"):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        host = cfg["server"]["host"]
        port = cfg["server"]["port"]
        self.server_endpoint = f"tcp://{host}:{port}"
        self.heartbeat_interval = cfg.get("heartbeat_interval", 5)

        # Derive ID from hostname
        hostname = socket.gethostname()
        m = re.match(r"rpi-(.+)", hostname)
        if not m:
            raise ValueError(f"Hostname '{hostname}' does not match expected pattern 'rpi-<ID>'")
        self.client_id = m.group(1).encode()

        # State
        self.running = False
        self.thread = None

        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.client_id)

        # Robust reconnection handling
        self.socket.setsockopt(zmq.RECONNECT_IVL, 1000)        # retry every 1s
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)    # up to 5s backoff
        self.socket.setsockopt(zmq.HEARTBEAT_IVL, 3000)        # client heartbeats to server
        self.socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 10000)
        self.socket.setsockopt(zmq.HEARTBEAT_TTL, 30000)

    def start(self):
        if self.running:
            return
        self.running = True

        # Connect (non-blocking even if server is DOWN)
        self.socket.connect(self.server_endpoint)

        # Launch background thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        try:
            self.socket.close(0)
        except:
            pass
        try:
            self.context.term()
        except:
            pass

    def join(self):
        if self.thread:
            self.thread.join()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        last_heartbeat = 0

        while self.running:
            now = time.time()

            # Send heartbeat (non-blocking)
            if now - last_heartbeat >= self.heartbeat_interval:
                try:
                    self.socket.send_multipart([b"heartbeat", b"alive"], zmq.NOBLOCK)
                except zmq.Again:
                    # Server not reachable yet — this is fine
                    pass
                last_heartbeat = now

            # Poll server messages
            try:
                events = dict(poller.poll(timeout=500))
            except zmq.error.ZMQError:
                break

            if self.socket in events:
                try:
                    frames = self.socket.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    break

                self._handle_server_message(frames)

        # Cleanup
        try:
            self.socket.close(0)
        except:
            pass

    def _handle_server_message(self, frames):
        if not frames:
            return

        command = frames[0].decode()

        if command == "ping":
            try:
                self.socket.send_multipart([b"pong", b"ok"], zmq.NOBLOCK)
            except zmq.Again:
                pass

        elif command == "do_something":
            result = b"done"
            try:
                self.socket.send_multipart([b"result", result], zmq.NOBLOCK)
            except zmq.Again:
                pass

        else:
            try:
                self.socket.send_multipart([b"error", b"unknown_command"], zmq.NOBLOCK)
            except zmq.Again:
                pass
