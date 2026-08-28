import codecs
import importlib.util
import pickle
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlonmcu.session import rpc


CLOUDPICKLE_AVAILABLE = importlib.util.find_spec("cloudpickle") is not None
requires_cloudpickle = pytest.mark.skipif(
    not CLOUDPICKLE_AVAILABLE,
    reason="RPC execution requires the optional cloudpickle package",
)


def test_remote_config_parses_tracker_address():
    config = rpc.RemoteConfig(tracker="tracker.example:9123", key="board")
    assert config.tracker_host == "tracker.example"
    assert config.tracker_port == 9123
    assert config.key == "board"


def test_rpc_session_connects_and_closes(monkeypatch):
    sock = MagicMock()
    monkeypatch.setattr(rpc.socket, "socket", MagicMock(return_value=sock))
    session = rpc.RPCSession("server", 9001, key="board", session_timeout=5)
    sock.connect.assert_called_once_with(("server", 9001))
    assert (session.url, session.port, session.key, session.session_timeout) == ("server", 9001, "board", 5)
    session.close()
    sock.close.assert_called_once()
    assert session._sock is None
    session.close()


def make_rpc_session():
    session = object.__new__(rpc.RPCSession)
    session._sock = MagicMock()
    session.url = "server"
    session.port = 9001
    session.key = "board"
    session.session_timeout = 0
    return session


def encode_result(value):
    return codecs.encode(pickle.dumps(value), "base64").decode("utf-8")


@requires_cloudpickle
def test_rpc_session_execute_round_trip(monkeypatch):
    session = make_rpc_session()
    sendjson = MagicMock()
    monkeypatch.setattr(rpc.base, "sendjson", sendjson)
    monkeypatch.setattr(rpc.base, "recvjson", MagicMock(return_value={"success": True, "results": [encode_result(42)]}))

    assert session.execute([{"model": "demo"}], until="run", parallel=2) == [42]
    message = sendjson.call_args.args[1]
    assert message["operation"] == "execute"
    assert message["until"] == "run"
    assert message["parallel"] == 2
    assert len(message["run_initializers"]) == 1


@pytest.mark.parametrize(
    "response",
    [None, {}, {"success": True}, {"success": False, "results": []}],
)
@requires_cloudpickle
def test_rpc_session_execute_rejects_invalid_responses(monkeypatch, response):
    session = make_rpc_session()
    monkeypatch.setattr(rpc.base, "sendjson", MagicMock())
    monkeypatch.setattr(rpc.base, "recvjson", MagicMock(return_value=response))
    with pytest.raises(AssertionError):
        session.execute([], until="run")


def test_unimplemented_rpc_file_operations():
    session = make_rpc_session()
    with pytest.raises(NotImplementedError):
        session.upload(b"data")
    with pytest.raises(NotImplementedError):
        session.download("file")
    with pytest.raises(NotImplementedError):
        session.remove("file")
    with pytest.raises(NotImplementedError):
        session.listdir(".")


def test_tracker_session_connect_close_and_free_server(monkeypatch):
    sock = MagicMock()
    connect_with_retry = MagicMock(return_value=sock)
    sendjson = MagicMock()
    monkeypatch.setattr(rpc.base, "connect_with_retry", connect_with_retry)
    monkeypatch.setattr(rpc.base, "sendjson", sendjson)
    tracker = rpc.TrackerSession(("tracker", 9000))
    connect_with_retry.assert_called_once_with(("tracker", 9000), timeout=10)

    tracker.free_server(SimpleNamespace(key="board", url="server", port=9001))
    assert sendjson.call_args.args[1] == {
        "action": "update_status",
        "key": "board",
        "addr": ["server", 9001],
        "status": "free",
    }
    tracker.close()
    sock.close.assert_called_once()


def test_tracker_request_server(monkeypatch):
    tracker = object.__new__(rpc.TrackerSession)
    tracker._addr = ("tracker", 9000)
    tracker._sock = MagicMock()
    monkeypatch.setattr(rpc.base, "sendjson", MagicMock())
    monkeypatch.setattr(rpc.base, "recvjson", MagicMock(return_value={"server_address": ["server", 9001]}))
    connect = MagicMock(return_value="session")
    monkeypatch.setattr(rpc, "connect", connect)

    assert tracker.request_server("board", session_timeout=3) == "session"
    connect.assert_called_once_with("server", 9001, "board", 3)


def test_tracker_request_server_reconnects_then_fails(monkeypatch):
    tracker = object.__new__(rpc.TrackerSession)
    tracker._addr = ("tracker", 9000)
    tracker._sock = MagicMock()
    monkeypatch.setattr(rpc.base, "sendjson", MagicMock(side_effect=socket.error("lost")))
    reconnect = MagicMock(side_effect=lambda: setattr(tracker, "_sock", MagicMock()))
    monkeypatch.setattr(tracker, "_connect", reconnect)

    with pytest.raises(RuntimeError, match="Cannot request board after 2 retry"):
        tracker.request_server("board", max_retry=2)
    reconnect.assert_called_once()


def test_connect_helpers_construct_sessions(monkeypatch):
    session = MagicMock()
    session_type = MagicMock(return_value=session)
    monkeypatch.setattr(rpc, "RPCSession", session_type)
    assert rpc.connect("server", 9001, "board", 4) is session
    session_type.assert_called_once_with("server", 9001, key="board", session_timeout=4)

    tracker = MagicMock()
    tracker_type = MagicMock(return_value=tracker)
    monkeypatch.setattr(rpc, "TrackerSession", tracker_type)
    assert rpc._connect_tracker("tracker", 9000) is tracker
    tracker_type.assert_called_once_with(("tracker", 9000))


def test_connect_tracker_checked_and_unchecked(monkeypatch):
    tracker = MagicMock()
    connect_tracker = MagicMock(return_value=tracker)
    monkeypatch.setattr(rpc, "_connect_tracker", connect_tracker)
    assert rpc.connect_tracker("tracker", 9000) is tracker
    assert rpc.connect_tracker("tracker", 9000, check=True) is tracker


def test_connect_tracker_checked_timeout(monkeypatch):
    thread = MagicMock()
    thread.is_alive.return_value = True
    monkeypatch.setattr(rpc, "Thread", MagicMock(return_value=thread))
    with pytest.raises(ValueError, match="Unable to connect to the tracker"):
        rpc.connect_tracker("tracker", 9000, timeout_sec=0.1, check=True)
    thread.start.assert_called_once()
    thread.join.assert_called_once_with(0.1)
