"""Benchmark the overhead of the MLonMCU remote execution (RPC).

1. transport: tracker + server on loopback with a no-op execute function (isolates connection setup, tracker
   matching and (de)serialization from the actual flow).
2. flow: wall time of `mlonmcu flow run` with the process_pool executor vs. the rpc executor (needs MLONMCU_HOME).

Usage: python scripts/bench_rpc.py [--skip-flow] [--model toycar] [--runs 1 4 8] [--parallel 4]
"""

import argparse
import os
import statistics
import subprocess
import sys
import time


def bench_transport(repeats=200, sizes=(1, 10, 100, 1000)):
    from mlonmcu.session.run import RunInitializer
    from mlonmcu.session.rpc import RemoteConfig
    from mlonmcu.session.rpc_server import RPCServer
    from mlonmcu.session.rpc_tracker import RPCTracker
    from mlonmcu.session.schedule import _process_rpc

    tracker = RPCTracker("127.0.0.1", 0, ping_interval=0).start()
    server = RPCServer(f"127.0.0.1:{tracker.port}", "bench", "127.0.0.1", 0, execute_fn=lambda i, u, p: list(i)).start()
    config = RemoteConfig(f"127.0.0.1:{tracker.port}", "bench")
    print("== transport (no-op server, loopback) ==")
    print(f"{'initializers':>12} {'median [ms]':>12} {'p95 [ms]':>10} {'per init [ms]':>14}")
    try:
        for size in sizes:
            inits = [
                RunInitializer(
                    idx=i,
                    model_name="toycar",
                    backend_name="tflmi",
                    target_name="host_x86",
                    platform_names=["mlif"],
                    config={"toycar.x": i, "mlif.print_outputs": True},
                )
                for i in range(size)
            ]
            n = repeats if size <= 100 else max(10, repeats // 10)
            times = []
            for _ in range(n):
                start = time.perf_counter()
                _process_rpc(inits, 3, 1, config)
                times.append((time.perf_counter() - start) * 1000)
            times.sort()
            med = statistics.median(times)
            print(f"{size:>12} {med:>12.2f} {times[int(0.95 * (len(times) - 1))]:>10.2f} {med / size:>14.3f}")
    finally:
        server.stop()
        tracker.stop()


def flow(model, runs, extra):
    cmd = [sys.executable, "-m", "mlonmcu.cli.main", "flow", "run", model, "--target", "host_x86", "--backend", "tflmi"]
    cmd += ["--config-gen", "_"] * runs
    cmd += extra
    start = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    ok = out.returncode == 0 and "All runs completed successfuly" in out.stdout + out.stderr
    return elapsed, ok


def bench_flow(model, run_counts, parallel, repeats):
    from mlonmcu.session.rpc_server import RPCServer
    from mlonmcu.session.rpc_tracker import RPCTracker

    assert os.environ.get("MLONMCU_HOME"), "MLONMCU_HOME is required for the flow benchmark"
    tracker = RPCTracker("127.0.0.1", 0).start()
    server = RPCServer(f"127.0.0.1:{tracker.port}", "bench", "127.0.0.1", 0).start()
    rpc = ["-c", "session.executor=rpc", "-c", f"session.rpc_tracker=127.0.0.1:{tracker.port}"]
    rpc += ["-c", "session.rpc_key=bench"]
    print(f"\n== flow ({model}, host_x86/tflmi, parallel={parallel}, wall time incl. CLI startup) ==")
    print(f"{'runs':>5} {'executor':>22} {'median [s]':>11} {'overhead [s]':>13}")
    try:
        flow(model, 1, ["--parallel", "1"])  # warm up caches/builds
        for n in run_counts:
            base = None
            variants = [
                ("process_pool", ["--parallel", str(parallel), "-c", "session.executor=process_pool"]),
                (
                    "rpc (1 batch)",
                    [
                        "--parallel",
                        "1",
                        "-c",
                        f"session.batch_size={n}",
                        "-c",
                        f"session.parallel_jobs={parallel}",
                        *rpc,
                    ],
                ),
                ("rpc (batch=1)", ["--parallel", "1", "-c", "session.batch_size=1", *rpc]),
            ]
            for name, extra in variants:
                times = []
                for _ in range(repeats):
                    elapsed, ok = flow(model, n, extra)
                    if not ok:
                        print(f"{n:>5} {name:>22} FAILED")
                        break
                    times.append(elapsed)
                else:
                    med = statistics.median(times)
                    base = med if base is None else base
                    print(f"{n:>5} {name:>22} {med:>11.2f} {med - base:>+13.2f}", flush=True)
    finally:
        server.stop()
        tracker.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-transport", action="store_true")
    parser.add_argument("--model", default="toycar")
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.skip_transport:
        bench_transport()
    if not args.skip_flow:
        bench_flow(args.model, args.runs, args.parallel, args.repeats)


if __name__ == "__main__":
    main()
