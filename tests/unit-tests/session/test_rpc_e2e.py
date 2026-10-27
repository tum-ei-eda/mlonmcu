"""End-to-end tests of tracker, server and client over loopback sockets (execution is stubbed)."""

import threading
from types import SimpleNamespace
import time

import pytest

pytest.importorskip("cloudpickle")

from mlonmcu.session import rpc  # noqa: E402
from mlonmcu.session import schedule  # noqa: E402
from mlonmcu.session.rpc_server import RPCServer  # noqa: E402
from mlonmcu.session.rpc_tracker import RPCTracker  # noqa: E402


def echo(run_initializers, until, parallel):
    return [{"init": x, "until": int(until), "parallel": parallel} for x in run_initializers]


@pytest.fixture
def tracker():
    t = RPCTracker("127.0.0.1", 0).start()
    yield t
    t.stop()


def make_server(tracker, key="default", fn=echo):
    return RPCServer(f"127.0.0.1:{tracker.port}", key, "127.0.0.1", 0, execute_fn=fn).start()


def wait_for(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def config(tracker, key="default"):
    return rpc.RemoteConfig(f"127.0.0.1:{tracker.port}", key)


def test_round_trip_and_server_is_freed(tracker):
    server = make_server(tracker)
    try:
        res = schedule._process_rpc(["a", "b"], 3, 2, config(tracker))
        assert res == [{"init": "a", "until": 3, "parallel": 2}, {"init": "b", "until": 3, "parallel": 2}]
        assert wait_for(lambda: tracker.state.summary()["data"]["server_info"][0]["status"] == "free")
        # A second session reuses the same server
        assert schedule._process_rpc(["c"], 1, 1, config(tracker))[0]["init"] == "c"
    finally:
        server.stop()


def test_remote_exception_is_propagated_and_server_recovers(tracker):
    calls = []

    def flaky(inits, until, parallel):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("kaboom")
        return echo(inits, until, parallel)

    server = make_server(tracker, fn=flaky)
    try:
        with pytest.raises(RuntimeError, match="kaboom"):
            schedule._process_rpc(["a"], 1, 1, config(tracker))
        assert wait_for(lambda: tracker.state.summary()["data"]["server_info"][0]["status"] == "free")
        assert schedule._process_rpc(["a"], 1, 1, config(tracker))[0]["init"] == "a"
    finally:
        server.stop()


def test_requests_queue_until_a_server_is_free(tracker):
    gate = threading.Event()

    def slow(inits, until, parallel):
        gate.wait(5)
        return echo(inits, until, parallel)

    server = make_server(tracker, fn=slow)
    results = []
    threads = [
        threading.Thread(target=lambda i=i: results.append(schedule._process_rpc([i], 1, 1, config(tracker))))
        for i in range(2)
    ]
    try:
        for t in threads:
            t.start()
        assert wait_for(lambda: tracker.state.summary()["data"]["queue_info"]["default"]["pending"] == 1)
        gate.set()
        for t in threads:
            t.join(10)
        assert sorted(r[0]["init"] for r in results) == [0, 1]
    finally:
        gate.set()
        server.stop()


def test_keys_are_matched_and_distribute_over_servers(tracker):
    a = make_server(tracker, "boardA")
    b = make_server(tracker, "boardB")
    try:
        assert schedule._process_rpc(["x"], 1, 1, config(tracker, "boardB"))[0]["init"] == "x"
        info = tracker.state.summary()["data"]["queue_info"]
        assert info["boardA"]["free"] == 1 and info["boardB"]["total"] == 1
    finally:
        a.stop()
        b.stop()


def test_request_without_server_times_out(tracker):
    t = rpc.connect_tracker("127.0.0.1", tracker.port)
    with pytest.raises(RuntimeError, match="No free server"):
        t.request_server("missing", request_timeout=0.2)
    t.close()


def test_unregistered_server_is_not_handed_out(tracker):
    server = make_server(tracker)
    server.stop()
    t = rpc.connect_tracker("127.0.0.1", tracker.port)
    with pytest.raises(RuntimeError, match="No free server"):
        t.request_server("default", request_timeout=0.2)
    t.close()


def test_dead_server_is_dropped_by_liveness_checks():
    t = RPCTracker("127.0.0.1", 0, ping_interval=0.1, max_failures=2).start()
    try:
        server = make_server(t)
        assert len(t.state.summary()["data"]["server_info"]) == 1
        server._srv.shutdown()
        server._srv.server_close()  # crash without unregistering
        assert wait_for(lambda: not t.state.summary()["data"]["server_info"])
    finally:
        t.stop()


def test_live_server_survives_liveness_checks():
    t = RPCTracker("127.0.0.1", 0, ping_interval=0.1, max_failures=2).start()
    server = make_server(t)
    try:
        time.sleep(0.8)
        assert len(t.state.summary()["data"]["server_info"]) == 1
    finally:
        server.stop()
        t.stop()


def test_session_postprocesses_are_applied_locally_for_initializers(tmp_path):
    from mlonmcu.session.run import RunInitializer
    from mlonmcu.report import Report

    report = Report()
    report.main_df = report.main_df.__class__({"Total ROM": [2000.0]})
    inits = [RunInitializer(idx=0, postprocess_names=["bytes2kb"], config={})]
    assert schedule._postprocess_pickable(inits, report, dest=tmp_path) == 0
    assert list(report.main_df.columns) == ["Total ROM [kB]"]
    assert report.main_df["Total ROM [kB]"][0] == 2.0


def make_artifact_fn(tmp_path):
    from mlonmcu.session.rpc_server import _pack_run_dirs

    def fn(inits, until, parallel, artifacts_to=None):
        results = []
        for i, init in enumerate(inits):
            d = tmp_path / f"server_run_{i}"
            (d / "sub").mkdir(parents=True)
            (d / "out.txt").write_text(f"result of {init.idx}")
            (d / "sub" / "big.bin").write_bytes(bytes(range(256)) * 4096)
            results.append(SimpleNamespace(idx=i, dir=d))
        if artifacts_to:
            _pack_run_dirs(results, artifacts_to)
        return results

    return fn


def test_artifacts_are_transferred_to_the_client(tracker, tmp_path):
    from mlonmcu.session.run import RunInitializer

    server = make_server(tracker, fn=make_artifact_fn(tmp_path))
    runs_dir = tmp_path / "client_runs"
    runs_dir.mkdir()
    inits = [RunInitializer(idx=5, config={}), RunInitializer(idx=7, config={})]
    try:
        cfg = rpc.RemoteConfig(f"127.0.0.1:{tracker.port}", "default", "all")
        results = schedule._process_rpc(inits, 3, 1, cfg, runs_dir=runs_dir)
        assert (runs_dir / "5" / "out.txt").read_text() == "result of 5"
        assert (runs_dir / "7" / "out.txt").read_text() == "result of 7"
        assert (runs_dir / "7" / "sub" / "big.bin").stat().st_size == 256 * 4096
        assert [r.dir for r in results] == [runs_dir / "5", runs_dir / "7"]
    finally:
        server.stop()


def test_artifacts_are_discarded_by_default(tracker, tmp_path):
    from mlonmcu.session.run import RunInitializer

    server = make_server(tracker, fn=make_artifact_fn(tmp_path))
    runs_dir = tmp_path / "client_runs"
    runs_dir.mkdir()
    try:
        cfg = rpc.RemoteConfig(f"127.0.0.1:{tracker.port}", "default")
        schedule._process_rpc([RunInitializer(idx=0, config={})], 3, 1, cfg, runs_dir=runs_dir)
        assert list(runs_dir.iterdir()) == []
    finally:
        server.stop()


def test_remote_config_rejects_unknown_artifact_mode():
    with pytest.raises(ValueError, match="rpc_artifacts"):
        rpc.RemoteConfig("t:1", "k", "some")


def test_extract_rejects_path_traversal_and_skips_escaping_links(tmp_path):
    import io
    import tarfile

    from mlonmcu.session.run import RunInitializer

    inits = [RunInitializer(idx=0, config={})]
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tar:
        info = tarfile.TarInfo("0/../../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError, match="Unexpected member"):
        schedule._extract_run_dirs(evil, inits, [], runs_dir)

    links = tmp_path / "links.tar"
    with tarfile.open(links, "w") as tar:
        info = tarfile.TarInfo("0/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    schedule._extract_run_dirs(links, inits, [], runs_dir)
    assert not (runs_dir / "0" / "link").exists() and not (runs_dir / "0" / "link").is_symlink()
    assert not (tmp_path / "escape.txt").exists()
