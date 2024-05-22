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
    def tracker_host(self):
        return self.tracker.split(":")[0]

    @property
    def tracker_port(self):
        return int(self.tracker.split(":")[1])


class RPCSession(object):
    """RPC Client session module

    Do not directly create the object, call connect
    """

    # def __init__(self, sess):
    #     self._sess = sess
    def __init__(self, url, port, key="", session_timeout=0):
        # self._sess = sess
        print("__init__")
        self.url = url
        self.port = port
        self.key = key
        self.session_timeout = session_timeout
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("_sock.connect")
        self._sock.connect((url, port))
        print("_sock.connected")

    def __del__(self):
        self.close()

    def close(self):
        """Close the server connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def execute(self, run_initializers: List[RunInitializer], until: RunStage, parallel: int = 1) -> RunResult:
        print("execute")
        # TODO: move imports
        import codecs
        import pickle
        import cloudpickle  # TODO: update requirements.txt
        run_initializers = [codecs.encode(cloudpickle.dumps(x), "base64").decode("utf8") for x in run_initializers]
        msg = {"operation": "execute", "run_initializers": run_initializers, "until": until, "parallel": parallel}
        print("msg", msg)
        assert self._sock is not None
        # TODO: pickle?
        base.sendjson(self._sock, msg)
        response = base.recvjson(self._sock)
        print("response", response)
        # <- {"results": [result0,...]}
        assert response is not None
        success = response.get("success", None)
        assert success is not None
        results = response.get("results", None)
        assert results is not None
        print("success", success)
        assert success, "Session failed!"
        print("r", results)
        results = [pickle.loads(codecs.decode(x.encode("utf-8"), "base64")) for x in results]
        print("results", results)
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
        timeout = 10
        self._sock = base.connect_with_retry(self._addr, timeout=timeout)
        # TODO: implement magic
        # self._sock.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
        # magic = struct.unpack("<i", base.recvall(self._sock, 4))[0]
        # if magic != base.RPC_TRACKER_MAGIC:
        #     raise RuntimeError(f"{str(self._addr)} is not RPC Tracker")

    def close(self):
        """Close the tracker connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def free_server(self, server):
        assert self._sock is not None
        base.sendjson(self._sock, {
            'action': 'update_status',
            'key': server.key,
            'addr': [server.url, server.port],
            'status': 'free'
        })
        # TODO: response?

    def request_server(
        self, key, priority=1, session_timeout=0, max_retry=5
    ):
        print("request_server", key, priority, session_timeout, max_retry)
        # TODO: implement priority
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
        print("for")
        for _ in range(max_retry):
            print("try")
            try:
                if self._sock is None:
                    print("_connect")
                    self._connect()
                print("connected")
                # base.sendjson(self._sock, [base.TrackerCode.REQUEST, key, "", priority])
                base.sendjson(self._sock, {'action': 'request_server', 'key': key})
                print("requested")
                # value = base.recvjson(self._sock)
                server_info = base.recvjson(self._sock)
                print("received")
                assert server_info
                # if value[0] != base.TrackerCode.SUCCESS:
                #     raise RuntimeError(f"Invalid return value {str(value)}")
                # url, port, matchkey = value[1]
                server_address = server_info.get('server_address')
                print("server_address", server_address)
                assert server_address
                url, port = server_address
                print("connect to server")
                return connect(
                    url,
                    port,
                    # matchkey,
                    key,
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
    # sess = None  # TODO
    # return RPCSession(sess)
    print("connect", url, port, key)
    return RPCSession(url, port, key=key, session_timeout=session_timeout)


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


def connect_tracker(tracker_host: str, tracker_port: int, timeout_sec=10, check=False):  # TODO: update timeout
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
