#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""MLonMCU RPC tracker: matches clients with free RPC servers by key.

Protocol (length-prefixed JSON, see rpc_utils.sendjson/recvjson):

* ``{"action": "register", "key": K, "addr": [host, port]}`` -> ``{"success": true}``
* ``{"action": "request_server", "key": K, "timeout": seconds?}`` -> ``{"server_address": [host, port]}``
  (blocks until a server is free; ``{"error": ...}`` on timeout)
* ``{"action": "update_status", "key": K, "addr": [host, port], "status": "free"|"busy"}`` (no reply)
* ``{"action": "unregister", "key": K, "addr": [host, port]}`` -> ``{"success": true}``
* ``{"action": "summary"}`` -> ``{"code": "success", "data": {"server_info": [...], "queue_info": {...}}}``
  (format understood by the original ``query_tracker.py``)

Servers are pinged (``{"message": "ping"}`` -> ``{"message": "pong"}``) every ``ping_interval`` seconds and are
dropped after ``max_failures`` consecutive failures.
"""

import argparse
import logging
import socketserver
import threading
from collections import defaultdict
from typing import Optional

import socket
import mlonmcu.session.rpc_utils as base

logger = logging.getLogger("RPCTracker")


class TrackerState:
    """Thread-safe bookkeeping of registered servers and waiting clients."""

    def __init__(self):
        self._cond = threading.Condition()
        self._servers = defaultdict(dict)  # key -> {(host, port): free}
        self._pending = defaultdict(int)

    def register(self, key, addr):
        with self._cond:
            self._servers[key][tuple(addr)] = True
            self._cond.notify_all()

    def unregister(self, key, addr):
        with self._cond:
            self._servers[key].pop(tuple(addr), None)

    def set_status(self, key, addr, free: bool):
        with self._cond:
            addr = tuple(addr)
            if addr in self._servers[key]:
                self._servers[key][addr] = free
                if free:
                    self._cond.notify_all()

    def acquire(self, key, timeout: Optional[float] = None):
        """Block until a server with `key` is free and mark it busy. Returns None on timeout."""
        with self._cond:
            self._pending[key] += 1
            try:
                ok = self._cond.wait_for(lambda: any(self._servers[key].values()), timeout=timeout)
                if not ok:
                    return None
                for addr, free in self._servers[key].items():
                    if free:
                        self._servers[key][addr] = False
                        return addr
            finally:
                self._pending[key] -= 1
        return None

    def summary(self):
        with self._cond:
            keys = set(self._servers) | {k for k, n in self._pending.items() if n}
            return {
                "code": "success",
                "data": {
                    "server_info": [
                        {"addr": list(a), "key": k, "status": "free" if f else "busy"}
                        for k, v in self._servers.items()
                        for a, f in v.items()
                    ],
                    "queue_info": {
                        k: {
                            "total": len(self._servers[k]),
                            "free": sum(self._servers[k].values()),
                            "pending": self._pending[k],
                        }
                        for k in keys
                    },
                },
            }

    def servers(self):
        with self._cond:
            return [(k, a) for k, v in self._servers.items() for a in v]


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        state: TrackerState = self.server.state
        sock = self.request
        while True:
            try:
                msg = base.recvjson(sock)
            except (IOError, ValueError):
                break
            action = msg.get("action")
            key = msg.get("key")
            if action == "register":
                state.register(key, msg["addr"])
                base.sendjson(sock, {"success": True})
            elif action == "unregister":
                state.unregister(key, msg["addr"])
                base.sendjson(sock, {"success": True})
            elif action == "request_server":
                addr = state.acquire(key, timeout=msg.get("timeout"))
                if addr is None:
                    base.sendjson(sock, {"error": f"No free server for key '{key}'"})
                else:
                    try:
                        base.sendjson(sock, {"server_address": list(addr)})
                    except OSError:
                        state.set_status(key, addr, True)
                        break
            elif action == "update_status":
                state.set_status(key, msg["addr"], msg.get("status") == "free")
            elif action == "summary":
                base.sendjson(sock, state.summary())
            else:
                base.sendjson(sock, {"error": f"Unknown action '{action}'"})


class _ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RPCTracker:
    """Runs the tracker in a background thread (use `port=0` to pick a free port)."""

    def __init__(self, host="0.0.0.0", port=9000, ping_interval=5.0, max_failures=3):
        self.ping_interval = ping_interval
        self.max_failures = max_failures
        self._stop = threading.Event()
        self.state = TrackerState()
        self._srv = _ThreadedServer((host, port), _Handler)
        self._srv.state = self.state
        self.host = host
        self.port = self._srv.server_address[1]
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        self._start_monitor()
        return self

    def serve_forever(self):
        self._start_monitor()
        self._srv.serve_forever()

    def _start_monitor(self):
        if self.ping_interval and self.ping_interval > 0:
            threading.Thread(target=self._monitor, daemon=True).start()

    def _ping(self, addr):
        try:
            with socket.create_connection(addr, timeout=2) as sock:
                sock.settimeout(2)
                base.sendjson(sock, {"message": "ping"})
                return (base.recvjson(sock) or {}).get("message") == "pong"
        except (OSError, ValueError):
            return False

    def _monitor(self):
        failures = {}
        while not self._stop.wait(self.ping_interval):
            alive = set()
            for key, addr in self.state.servers():
                alive.add((key, addr))
                if self._ping(addr):
                    failures.pop((key, addr), None)
                    continue
                failures[(key, addr)] = failures.get((key, addr), 0) + 1
                if failures[(key, addr)] >= self.max_failures:
                    logger.warning("Server %s (%s) unreachable, unregistering", addr, key)
                    self.state.unregister(key, addr)
                    failures.pop((key, addr))
            failures = {k: v for k, v in failures.items() if k in alive}

    def stop(self):
        self._stop.set()
        self._srv.shutdown()
        self._srv.server_close()
        if self._thread:
            self._thread.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLonMCU RPC tracker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ping-interval", type=float, default=5.0, help="seconds, 0 disables liveness checks")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    tracker = RPCTracker(args.host, args.port, args.ping_interval)
    logger.info("Tracker listening on %s:%d", args.host, tracker.port)
    try:
        tracker.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
