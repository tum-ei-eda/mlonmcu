import errno
import json
import socket
import struct
from unittest.mock import MagicMock

import pytest

from mlonmcu.session import rpc_utils


class ReceivingSocket:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.sizes = []

    def recv(self, size):
        self.sizes.append(size)
        return next(self.chunks)


def test_py_str_and_address_family(monkeypatch):
    assert rpc_utils.py_str(b"hello") == "hello"
    getaddrinfo = MagicMock(return_value=[(socket.AF_INET6, None, None, None, None)])
    monkeypatch.setattr(rpc_utils.socket, "getaddrinfo", getaddrinfo)
    assert rpc_utils.get_addr_family(("localhost", 9000)) == socket.AF_INET6
    getaddrinfo.assert_called_once_with("localhost", 9000, 0, 0, socket.IPPROTO_TCP)


def test_recvall_combines_partial_reads_and_caps_chunk_size():
    sock = ReceivingSocket([b"a" * 1024, b"bc"])
    assert rpc_utils.recvall(sock, 1026) == b"a" * 1024 + b"bc"
    assert sock.sizes == [1024, 2]


def test_recvall_reports_closed_connection():
    with pytest.raises(IOError, match="connection reset"):
        rpc_utils.recvall(ReceivingSocket([b""]), 1)


def test_sendjson_writes_length_prefixed_utf8():
    sock = MagicMock()
    value = {"message": "ümlaut", "count": 2}
    encoded = json.dumps(value).encode("utf-8")

    rpc_utils.sendjson(sock, value)

    assert sock.sendall.call_args_list[0].args[0] == struct.pack("<i", len(encoded))
    assert sock.sendall.call_args_list[1].args[0] == encoded


def test_recvjson_reads_length_prefixed_utf8():
    encoded = json.dumps({"success": True, "items": [1, 2]}).encode("utf-8")
    sock = ReceivingSocket([struct.pack("<i", len(encoded)), encoded])
    assert rpc_utils.recvjson(sock) == {"success": True, "items": [1, 2]}


def test_random_key_retries_collisions(monkeypatch):
    values = iter([0.1, 0.2])
    monkeypatch.setattr(rpc_utils.random, "random", lambda: next(values))
    assert rpc_utils.random_key("device", delimiter="-", cmap={"device-0.1": object()}) == "device-0.2"
    assert rpc_utils.split_random_key("device-0.2", delimiter="-") == ["device", "0.2"]


def test_connect_with_retry_succeeds(monkeypatch):
    sock = MagicMock()
    monkeypatch.setattr(rpc_utils, "get_addr_family", lambda addr: socket.AF_INET)
    socket_factory = MagicMock(return_value=sock)
    monkeypatch.setattr(rpc_utils.socket, "socket", socket_factory)

    assert rpc_utils.connect_with_retry(("localhost", 9000)) is sock
    socket_factory.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect.assert_called_once_with(("localhost", 9000))


def test_connect_with_retry_retries_refused_connection(monkeypatch):
    first = MagicMock()
    first.connect.side_effect = socket.error(errno.ECONNREFUSED, "refused")
    second = MagicMock()
    monkeypatch.setattr(rpc_utils.socket, "socket", MagicMock(side_effect=[first, second]))
    monkeypatch.setattr(rpc_utils, "get_addr_family", lambda addr: socket.AF_INET)
    sleep = MagicMock()
    monkeypatch.setattr(rpc_utils.time, "sleep", sleep)
    monkeypatch.setattr(rpc_utils.time, "time", MagicMock(side_effect=[0, 0.5, 0.5]))

    assert rpc_utils.connect_with_retry(("localhost", 9000), timeout=1, retry_period=0.25) is second
    sleep.assert_called_once_with(0.25)


def test_connect_with_retry_times_out(monkeypatch):
    sock = MagicMock()
    sock.connect.side_effect = socket.error(errno.ECONNREFUSED, "refused")
    monkeypatch.setattr(rpc_utils.socket, "socket", MagicMock(return_value=sock))
    monkeypatch.setattr(rpc_utils, "get_addr_family", lambda addr: socket.AF_INET)
    monkeypatch.setattr(rpc_utils.time, "time", MagicMock(side_effect=[0, 2]))

    with pytest.raises(RuntimeError, match="Failed to connect"):
        rpc_utils.connect_with_retry(("localhost", 9000), timeout=1)


def test_connect_with_retry_propagates_other_socket_errors(monkeypatch):
    sock = MagicMock()
    error = socket.error(errno.ENETUNREACH, "unreachable")
    sock.connect.side_effect = error
    monkeypatch.setattr(rpc_utils.socket, "socket", MagicMock(return_value=sock))
    monkeypatch.setattr(rpc_utils, "get_addr_family", lambda addr: socket.AF_INET)

    with pytest.raises(socket.error) as exc_info:
        rpc_utils.connect_with_retry(("localhost", 9000))
    assert exc_info.value is error
