"""Real session over RPC (tracker + server + `mlonmcu flow run`) using the host_x86/tflmi flow.

Requires an initialized MLONMCU_HOME with the tflite-micro dependencies and the `toycar` and `aww` models.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("cloudpickle")

from mlonmcu.session.rpc_server import RPCServer  # noqa: E402
from mlonmcu.session.rpc_tracker import RPCTracker  # noqa: E402

HOME = os.environ.get("MLONMCU_HOME")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HOME or not os.path.isdir(HOME), reason="MLONMCU_HOME is not set"),
]


def flow(*extra):
    cmd = [sys.executable, "-m", "mlonmcu.cli.main", "flow", "run", "toycar", "--target", "host_x86"]
    cmd += ["--backend", "tflmi", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)


def test_rpc_flow_matches_local_run():
    tracker = RPCTracker("127.0.0.1", 0).start()
    server = RPCServer(f"127.0.0.1:{tracker.port}", "itest", "127.0.0.1", 0).start()
    try:
        local = flow()
        assert local.returncode == 0, local.stdout + local.stderr
        remote = flow(
            "-c", "session.executor=rpc",
            "-c", f"session.rpc_tracker=127.0.0.1:{tracker.port}",
            "-c", "session.rpc_key=itest",
        )  # fmt: skip
        assert remote.returncode == 0, remote.stdout + remote.stderr
        for out in (local, remote):
            assert "All runs completed successfuly" in out.stdout + out.stderr

        def rom(out):
            line = [ln for ln in out.stdout.splitlines() if ln.startswith("0 ")][0]
            return line.split()[line.split().index("True") + 1]

        assert rom(local) == rom(remote)
    finally:
        server.stop()
        tracker.stop()


def test_rpc_batches_keep_run_indices_and_apply_session_postprocesses():
    tracker = RPCTracker("127.0.0.1", 0).start()
    server = RPCServer(f"127.0.0.1:{tracker.port}", "itest", "127.0.0.1", 0).start()
    try:
        cmd = [sys.executable, "-m", "mlonmcu.cli.main", "flow", "run", "toycar", "aww", "--target", "host_x86"]
        cmd += ["--backend", "tflmi", "--postprocess", "bytes2kb", "-c", "session.executor=rpc"]
        cmd += ["-c", f"session.rpc_tracker=127.0.0.1:{tracker.port}", "-c", "session.rpc_key=itest"]
        cmd += ["-c", "session.batch_size=1"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
        text = out.stdout + out.stderr
        assert out.returncode == 0, text
        assert "not supported in pickable mode" not in text
        rows = [ln.split() for ln in out.stdout.splitlines() if ln[:2].strip().isdigit()]
        assert [r[1] for r in rows] == ["0", "1"]
        assert "Total ROM [kB]" in out.stdout.replace("\n", " ") or "ROM [kB]" in out.stdout
    finally:
        server.stop()
        tracker.stop()


@pytest.mark.parametrize("mode", ["all", "none"])
def test_rpc_artifact_transfer(mode, tmp_path):
    tracker = RPCTracker("127.0.0.1", 0).start()
    server = RPCServer(f"127.0.0.1:{tracker.port}", "itest", "127.0.0.1", 0).start()
    dest = tmp_path / "session"
    try:
        out = flow(
            "-c", "session.executor=rpc",
            "-c", f"session.rpc_tracker=127.0.0.1:{tracker.port}",
            "-c", "session.rpc_key=itest",
            "-c", f"session.rpc_artifacts={mode}",
            "--dest", str(dest),
        )  # fmt: skip
        assert out.returncode == 0, out.stdout + out.stderr
        run_dir = dest / "runs" / "0"
        if mode == "all":
            names = {p.name for p in run_dir.rglob("*") if p.is_file()}
            assert "artifacts.yml" in names
            assert any(n.endswith(".tflite") or n.endswith(".elf") or n == "generic_mlonmcu" for n in names), names
        else:
            assert not list(run_dir.glob("*")) if run_dir.exists() else True
    finally:
        server.stop()
        tracker.stop()
