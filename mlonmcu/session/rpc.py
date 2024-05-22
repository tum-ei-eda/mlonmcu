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
"""Definition of MLonMCU rpc utilities."""
import struct
import socket
from dataclasses import dataclass
from typing import Optional, List
from threading import Thread

from mlonmcu.session.run import RunInitializer, RunResult, RunStage
import mlonmcu.session.rpc_utils as base


@dataclass
class RemoteConfig:
    tracker: str = "localhost:9000"
    key: str = "default"

    @property
    def tracker_hostname(self):
        return self.tracker.split(":")[0]

    @property
    def tracker_port(self):
        return int(self.tracker.split(":")[1])


class RPCSession(object):
    """RPC Client session module

    Do not directly create the object, call connect
    """

    def __init__(self, sess):
        self._sess = sess

    def execute(self, run_initializers: List[RunInitializer], until: RunStage, parallel: int = 1) -> RunResult:
        msg = {"action": "execute", "initializers": run_initializers, "until": until, "parallel": parallel}
        # TODO: pickle?
        base.sendjson(self._sock, msg)
        response = base.recvjson(self._sock)
        # <- {"results": [result0,...]}
        results = response["results"]
        return results

    def upload(self, data, target=None):
        """Upload file to remote runtime temp folder

        Parameters
        ----------
        data : str or bytearray
            The file name or binary in local to upload.

        target : str, optional
            The path in remote
        """
        raise NotImplementedError

    def download(self, path):
        """Download file from remote temp folder.

        Parameters
        ----------
        path : str
            The relative location to remote temp folder.

        Returns
        -------
        blob : bytearray
            The result blob from the file.
        """
        raise NotImplementedError

    def remove(self, path):
        """Remove file from remote temp folder.

        Parameters
        ----------
        path: str
            The relative location to remote temp folder.
        """
        raise NotImplementedError

    def listdir(self, path):
        """ls files from remote temp folder.

        Parameters
        ----------
        path: str
            The relative location to remote temp folder.

        Returns
        -------
        dirs: str
            The files in the given directory with split token ','.
        """
        raise NotImplementedError


class TrackerSession:
    """Tracker client session.

    Parameters
    ----------
    addr : tuple
        The address tuple
    """

    def __init__(self, addr):
        self._addr = addr
        self._sock = None
        self._connect()

    def __del__(self):
        self.close()

    def _connect(self):
        self._sock = base.connect_with_retry(self._addr)
        self._sock.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
        magic = struct.unpack("<i", base.recvall(self._sock, 4))[0]
        if magic != base.RPC_TRACKER_MAGIC:
            raise RuntimeError(f"{str(self._addr)} is not RPC Tracker")

    def close(self):
        """Close the tracker connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def request(
        self, key, priority=1, session_timeout=0, max_retry=5
    ):
        """Request a new connection from the tracker.

        Parameters
        ----------
        key : str
            The type key of the device.

        priority : int, optional
            The priority of the request.

        session_timeout : float, optional
            The duration of the session, allows server to kill
            the connection when duration is longer than this value.
            When duration is zero, it means the request must always be kept alive.

        max_retry : int, optional
            Maximum number of times to retry before give up.
        """
        last_err = None
        for _ in range(max_retry):
            try:
                if self._sock is None:
                    self._connect()
                base.sendjson(self._sock, [base.TrackerCode.REQUEST, key, "", priority])
                value = base.recvjson(self._sock)
                if value[0] != base.TrackerCode.SUCCESS:
                    raise RuntimeError(f"Invalid return value {str(value)}")
                url, port, matchkey = value[1]
                return connect(
                    url,
                    port,
                    matchkey,
                    session_timeout,
                )
            except socket.error as err:
                self.close()
                last_err = err
            # except TVMError as err:
            #     last_err = err
        raise RuntimeError(
            f"Cannot request {key} after {max_retry} retry, last_error:{str(last_err)}"
        )


def connect(
    url, port, key="", session_timeout=0,
):
    """Connect to RPC Server

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    key : str, optional
        Additional key to match server

    session_timeout : float, optional
        The duration of the session in seconds, allows server to kill
        the connection when duration is longer than this value.
        When duration is zero, it means the request must always be kept alive.

    Returns
    -------
    sess : RPCSession
        The connected session.

    Examples
    --------
    Normal usage
    .. code-block:: python

        client = rpc.connect(server_url, server_port, server_key)

    """
    # sess = _ffi_api.Connect(url, port, key, enable_logging, *session_constructor_args)
    sess = None  # TODO
    return RPCSession(sess)


def _connect_tracker(url, port):
    """Connect to a RPC tracker

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    Returns
    -------
    sess : TrackerSession
        The connected tracker session.
    """
    return TrackerSession((url, port))


def connect_tracker(tracker_host: str, tracker_port: int, timeout_sec=1, check=False):
    tracker: Optional[TrackerSession] = None

    def _connect():
        nonlocal tracker
        tracker = _connect_tracker(tracker_host, tracker_port)
    if check:
        t = Thread(target=_connect)
        t.start()
        t.join(timeout_sec)
        if t.is_alive() or tracker is None:
            raise ValueError(
                "Unable to connect to the tracker using the following configuration:\n"
                f"    tracker host: {tracker_host}\n"
                f"    tracker port: {tracker_port}\n"
                f"    timeout (sec): {timeout_sec}\n"
            )
    else:
        _connect()
        assert tracker is not None
    return tracker
