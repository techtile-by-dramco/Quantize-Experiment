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
            experiment_settings = yaml.safe_load(f)

        server_settings = experiment_settings.get("server", "")
        host = server_settings.get("host", "")
        messaging_port = server_settings.get("messaging_port", "")
        sync_port = server_settings.get("sync_port", "")
        self.messaging_endpoint = f"tcp://{host}:{messaging_port}"
        self.sync_endpoint = f"tcp://{host}:{sync_port}"
        self.heartbeat_interval = experiment_settings.get("heartbeat_interval", 5)

        # Derive ID from hostname
        hostname = socket.gethostname()
        m = re.match(r"rpi-(.+)", hostname, re.IGNORECASE)
        if not m:
            raise ValueError(f"Hostname '{hostname}' does not match expected pattern 'rpi-<ID>'")
        self.client_id = m.group(1).encode()

        # State
        self.running = False
        self.thread = None

        # Setup ZMQ
        self.context = zmq.Context()
        self.messaging = self.context.socket(zmq.DEALER)
        self.messaging.setsockopt(zmq.IDENTITY, self.client_id)

        self.sync = self.context.socket(zmq.SUB)
        self.sync.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

        # Robust reconnection handling
        self.messaging.setsockopt(zmq.RECONNECT_IVL, 1000)        # retry every 1s
        self.messaging.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)    # up to 5s backoff
        self.messaging.setsockopt(zmq.HEARTBEAT_IVL, 3000)        # client heartbeats to server
        self.messaging.setsockopt(zmq.HEARTBEAT_TIMEOUT, 10000)
        self.messaging.setsockopt(zmq.HEARTBEAT_TTL, 30000)
        
        # Event handling
        self.callbacks = {}

    def start(self):
        if self.running:
            return
        self.running = True

        # Connect (non-blocking even if server is DOWN)
        self.messaging.connect(self.messaging_endpoint)
        self.sync.connect(self.sync_endpoint)

        # Launch background thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        try:
            self.messaging.close(0)
            self.sync.close(0)
        except:
            pass
        try:
            self.context.term()
        except:
            pass

    def join(self):
        if self.thread:
            self.thread.join()

    def on(self, command, func):
        """Register a callback for a given server command."""
        self.callbacks[command] = func

    def send(self, msg_type, *payload_frames):
        """
        Send a message to the server.

        Parameters
        ----------
        msg_type : str
            Message type string (e.g., 'heartbeat', 'request', etc.).
        payload_frames : optional list of bytes or str
            Additional frames to send after the message type.
        """

        frames = [msg_type.encode()]
        for frame in payload_frames:
            if isinstance(frame, str):
                frame = frame.encode()
            frames.append(frame)

        self.messaging.send_multipart(frames)

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.messaging, zmq.POLLIN)
        poller.register(self.sync, zmq.POLLIN)

        last_heartbeat = 0

        while self.running:
            now = time.time()

            # Send heartbeat (non-blocking)
            if now - last_heartbeat >= self.heartbeat_interval:
                try:
                    self.messaging.send_multipart([b"heartbeat", b"alive"], zmq.NOBLOCK)
                except zmq.Again:
                    # Server not reachable yet — this is fine
                    pass
                last_heartbeat = now

            # Poll server messages
            try:
                events = dict(poller.poll(timeout=100))
            except zmq.error.ZMQError:
                break

            if self.messaging in events:
                try:
                    frames = self.messaging.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    break

                self._handle_server_message(frames)
                
            if self.sync in events:
                try:
                    frames = self.sync.recv_multipart(zmq.NOBLOCK)
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
        args = [f.decode() for f in frames[1:]]

        # If a callback exists, call it
        if command in self.callbacks:
            try:
                self.callbacks[command](command, args)
            except Exception as e:
                print(f"Callback error for {command}: {e}")
            return

        # Default built-in handlers
        if command == "ping":
            try:
                self.socket.send_multipart([b"pong", b"ok"], zmq.NOBLOCK)
            except zmq.Again:
                pass
        else:
            try:
                self.socket.send_multipart([b"error", b"unknown_command"], zmq.NOBLOCK)
            except zmq.Again:
                pass

