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
"""MLonMCU RPC server: executes pickled run initializers on behalf of remote clients.

SECURITY: requests contain pickled objects, i.e. arbitrary code execution for anyone who can
connect. Only expose servers/trackers inside trusted networks (VPN, SSH tunnel).
"""

import argparse
import codecs
import logging
import pickle
import socket
import socketserver
import tarfile
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Callable, Optional

import mlonmcu.session.rpc_utils as base

logger = logging.getLogger("RPCServer")


def _pack_run_dirs(results, dest):
    with tarfile.open(dest, "w") as tar:
        for pos, res in enumerate(results):
            if res is not None and res.dir is not None and Path(res.dir).is_dir():
                tar.add(res.dir, arcname=str(pos))


def default_execute(run_initializers, until, parallel, artifacts_to=None):
    """Run the initializers using a fresh MLonMCU context on the server host.

    If artifacts_to is given, the run directories are written to this tar file (members named by run position)
    before the temporary session is cleaned up.
    """
    from mlonmcu.session.run import RunStage
    from mlonmcu.context.context import MlonMcuContext
    from mlonmcu.session.postprocess import get_postprocesses
    from mlonmcu.session.postprocess.postprocess import SessionPostprocess

    if isinstance(until, str):
        until = RunStage[until]
    else:
        until = RunStage(until)
    # Session postprocesses work on the merged report and are applied by the client.
    available = get_postprocesses()
    for run_initializer in run_initializers:
        if run_initializer.postprocess_names:
            run_initializer.postprocess_names = [
                n for n in run_initializer.postprocess_names if not issubclass(available[n], SessionPostprocess)
            ]
    config = {"session.use_init_stage": False, "session.executor": "process_pool"}
    with MlonMcuContext(deps_lock="read") as context:
        with context.get_session(resume=False, config=config) as session:
            for run_initializer in run_initializers:
                session.add_run(run_initializer, ignore_idx=True)
            success = session.process_runs(
                until=until,
                per_stage=False,
                print_report=False,
                num_workers=parallel,
                progress=False,
                context=context,
                export=True,
            )
            results = list(session.results)
            if artifacts_to:
                _pack_run_dirs(results, artifacts_to)
    if not success:
        raise RuntimeError("Some runs failed on the server")
    return results


def _notify_tracker(tracker_addr, message, expect_reply=False):
    sock = base.connect_with_retry(tracker_addr, timeout=10, retry_period=1)
    try:
        base.sendjson(sock, message)
        if expect_reply:
            return base.recvjson(sock)
    finally:
        sock.close()
    return None


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        srv: "RPCServer" = self.server.owner
        sock = self.request
        is_execute = False
        try:
            msg = base.recvjson(sock)
            if msg.get("message") == "ping":
                base.sendjson(sock, {"message": "pong"})
                return
            if msg.get("operation") != "execute":
                base.sendjson(sock, {"success": False, "error": f"Unknown operation '{msg.get('operation')}'"})
                return
            is_execute = True
            tmp = None
            try:
                initializers = [
                    pickle.loads(codecs.decode(x.encode("utf-8"), "base64")) for x in msg["run_initializers"]
                ]
                parallel = msg.get("parallel", 1)
                if msg.get("artifacts"):
                    tmp = tempfile.TemporaryDirectory()
                    archive = Path(tmp.name) / "artifacts.tar"
                    results = srv.execute_fn(initializers, msg["until"], parallel, artifacts_to=archive)
                else:
                    results = srv.execute_fn(initializers, msg["until"], parallel)
                import cloudpickle

                encoded = [codecs.encode(cloudpickle.dumps(r), "base64").decode("utf8") for r in results]
                response = {"success": True, "results": encoded}
                if tmp is not None and archive.is_file():
                    response["artifacts_size"] = archive.stat().st_size
                base.sendjson(sock, response)
                if "artifacts_size" in response:
                    base.sendfile(sock, archive)
            except Exception:  # pylint: disable=broad-except
                logger.exception("Execution failed")
                base.sendjson(sock, {"success": False, "results": [], "error": traceback.format_exc()})
            finally:
                if tmp is not None:
                    tmp.cleanup()
        except (IOError, ValueError):
            logger.warning("Client connection lost")
        finally:
            if is_execute:
                srv.release()


class _ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RPCServer:
    """Executes one session at a time; reports itself free to the tracker afterwards."""

    def __init__(
        self,
        tracker: Optional[str] = None,
        key: str = "default",
        host: str = "0.0.0.0",
        port: int = 0,
        advertise_host: Optional[str] = None,
        execute_fn: Callable = default_execute,
    ):
        self.key = key
        self.execute_fn = execute_fn
        self._srv = _ThreadedServer((host, port), _Handler)
        self._srv.owner = self
        self.port = self._srv.server_address[1]
        self.host = advertise_host or (host if host != "0.0.0.0" else socket.gethostname())
        self._tracker_addr = None
        if tracker:
            h, p = tracker.split(":")
            self._tracker_addr = (h, int(p))
        self._thread = None

    @property
    def addr(self):
        return [self.host, self.port]

    def start(self):
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        self.register()
        return self

    def serve_forever(self):
        self.register()
        try:
            self._srv.serve_forever()
        finally:
            self.unregister()

    def register(self):
        if self._tracker_addr:
            _notify_tracker(
                self._tracker_addr, {"action": "register", "key": self.key, "addr": self.addr}, expect_reply=True
            )

    def unregister(self):
        if self._tracker_addr:
            try:
                _notify_tracker(
                    self._tracker_addr, {"action": "unregister", "key": self.key, "addr": self.addr}, expect_reply=True
                )
            except (OSError, RuntimeError):
                logger.warning("Could not unregister from tracker")

    def release(self):
        """Tell the tracker we are free again (also covers clients that crashed)."""
        if self._tracker_addr:
            try:
                _notify_tracker(
                    self._tracker_addr,
                    {"action": "update_status", "key": self.key, "addr": self.addr, "status": "free"},
                )
            except (OSError, RuntimeError):
                logger.warning("Could not notify tracker")

    def stop(self):
        self.unregister()
        self._srv.shutdown()
        self._srv.server_close()
        if self._thread:
            self._thread.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLonMCU RPC server")
    parser.add_argument("--tracker", default="localhost:9000", help="host:port of the tracker")
    parser.add_argument("--key", default="default")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--log-level", default="INFO", help="log level of the MLonMCU sessions run by this server")
    parser.add_argument("--advertise-host", default=None, help="address clients should use to reach this server")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from mlonmcu.logging import set_log_level

    set_log_level(args.log_level.upper())
    server = RPCServer(args.tracker, args.key, args.host, args.port, args.advertise_host)
    logger.info("Server '%s' listening on port %d", args.key, server.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
