#!/usr/bin/env python3
"""Measure fresh MLonMCU flows, cache reuse, and overlapping stage intervals.

Run in the workspace's Python environment. Raw sessions live outside the repo;
JSONL measurements are appended after every sample and support resuming.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def interval_union(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        total += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = max(end, stop) if end is not None else stop
    return total


def cache_stats(cache):
    env = {key: value for key, value in os.environ.items() if not key.startswith("CCACHE_")}
    env.update(CCACHE_DIR=str(cache), CCACHE_CONFIGPATH=str(cache / "ccache.conf"))
    output = subprocess.check_output(["ccache", "--print-stats"], env=env, text=True)
    return {key: int(value) for key, value in (line.split() for line in output.splitlines())}


def number(row, key):
    value = row.get(key, row.get(f"_{key}_"))
    return float(value) if value is not None and value != "" else None


def sample(args, spec, seed=None):
    name = spec["name"]
    dest = args.raw / name
    if dest.exists():
        raise RuntimeError(f"Unrecorded destination exists: {dest}; inspect it before resuming")
    dest.mkdir(parents=True)
    if spec["cache"] == "warm":
        if seed is None:
            raise ValueError(f"No cache seed for {name}")
        shutil.copytree(seed / "ccache", dest / "ccache")
    if spec["cache"] != "off":
        cache = dest / "ccache"
        cache.mkdir(exist_ok=True)
        (cache / "ccache.conf").touch(exist_ok=True)
        before_cache = cache_stats(cache)
    else:
        before_cache = {}
    common_config = {
        **args.workspace_vars,
        "run.profile_stages": True,
        "run.export_optional": True,
        "mlif.num_threads": spec["threads"],
        "mlif.ccache": False,
        "ccache.mode": "session",
        "ccache.max_size": "2G",
        # Hold TVM/BLAS internal threading constant while varying run and make parallelism.
        "tvmaotplus.num_threads": 1,
        "session.ccache_cleanup": False,
    }
    runs = []
    for index, model in enumerate(spec["models"]):
        runs.append(
            {
                "idx": index,
                "model_name": model,
                "frontend_names": ["tflite"],
                "backend_name": spec["backend"],
                "platform_names": ["mlif"],
                "target_name": "host_x86",
                "feature_names": [] if spec["cache"] == "off" else ["ccache"],
                "config": {**common_config, **spec.get("run_configs", [{}] * len(spec["models"]))[index]},
            }
        )
    initializer = dest / "input.yml"
    initializer.write_text(yaml.safe_dump({"runs": runs}))
    command = [
        sys.executable,
        "-m",
        "mlonmcu.cli.main",
        "flow",
        "run",
        "--initializer",
        str(initializer),
        "--parallel",
        str(spec["parallel"]),
        "--dest",
        str(dest),
        "-l",
        f"ccache_bench_{name}",
        "-c",
        f"runs_per_stage={spec['per_stage']}",
        "session.use_init_stage=true",
        "print_report=false",
    ]
    if spec.get("parallel_per_stage"):
        command.extend(["--parallel-per-stage", spec["parallel_per_stage"]])
    env = {key: value for key, value in os.environ.items() if not key.startswith("CCACHE_")}
    for key in ("CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER"):
        env.pop(key, None)
    env.update(
        MLONMCU_HOME=str(args.home),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        MPLCONFIGDIR=str(args.raw / "matplotlib"),
    )
    print(f"START {name} {time.strftime('%H:%M:%S')} load={os.getloadavg()[0]:.2f}", flush=True)
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    load_before = os.getloadavg()
    started_utc = time.time()
    started = time.perf_counter()
    with (dest / "command.log").open("w") as log:
        process = subprocess.run(command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    measurement = {
        **spec,
        "command": command,
        "destination": str(dest),
        "started_utc": started_utc,
        "wall_s": elapsed,
        "cpu_user_s": cpu_after.ru_utime - cpu_before.ru_utime,
        "cpu_system_s": cpu_after.ru_stime - cpu_before.ru_stime,
        "load_before": load_before,
        "load_after": os.getloadavg(),
        "returncode": process.returncode,
    }
    if process.returncode:
        measurement["error"] = (dest / "command.log").read_text()[-10000:]
        return measurement
    with (dest / "report.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(spec["models"]) or any(row.get("Failing", "False") == "True" for row in rows):
        raise RuntimeError(f"Unexpected report in {dest}")
    per_run = []
    for row in rows:
        index = int(row["Run"])
        run_dir = dest / "runs" / str(index)
        intervals = {}
        for stage in ("Load", "Build", "Compile", "Run"):
            intervals[stage.lower()] = [number(row, f"{stage} Stage {which} Time [s]") for which in ("Start", "End")]
            if any(value is None for value in intervals[stage.lower()]):
                raise RuntimeError(f"Missing {stage} interval in {dest}")
        elf = run_dir / "mlif/bin/generic_mlonmcu"
        per_run.append(
            {
                "run": index,
                "model": row["Model"],
                "intervals": intervals,
                "compile_work_s": number(row, "Compile Stage Time [s]"),
                "build_work_s": number(row, "Build Stage Time [s]"),
                "source_bytes": sum(path.stat().st_size for path in run_dir.glob("*.c*")),
                "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
            }
        )
    measurement["runs"] = per_run
    for stage in ("load", "build", "compile", "run"):
        intervals = [run["intervals"][stage] for run in per_run]
        measurement[f"{stage}_sum_s"] = sum(stop - start for start, stop in intervals)
        measurement[f"{stage}_union_s"] = interval_union(intervals)
        measurement[f"{stage}_span_s"] = max(stop for _, stop in intervals) - min(start for start, _ in intervals)
    if before_cache:
        after_cache = cache_stats(dest / "ccache")
        counters = {key: after_cache.get(key, 0) - before_cache.get(key, 0) for key in after_cache}
        hits = counters.get("direct_cache_hit", 0) + counters.get("preprocessed_cache_hit", 0)
        misses = counters.get("cache_miss", 0)
        measurement.update(cache_counters=counters, cache_hits=hits, cache_misses=misses)
        measurement["hit_rate"] = hits / (hits + misses) if hits + misses else None
    print(
        f"DONE {name}: wall={elapsed:.2f}s compile sum={measurement['compile_sum_s']:.2f}s "
        f"union={measurement['compile_union_s']:.2f}s hits={measurement.get('cache_hits', '-')}",
        flush=True,
    )
    return measurement


def plan(args):
    for backend in args.backends:
        if args.suite == "scheduling":
            # Fresh caches expose redundant work when equivalent compilations race.
            models = ["sine_model", "aww", "vww", "resnet"]
            prime_group = f"parallel-prime-{backend}"
            for label, limits, threads in (
                ("build1-compile4", "BUILD=1,COMPILE=4,RUN=4", 1),
                ("build4-compile1", "BUILD=4,COMPILE=1,RUN=4", 4),
                ("build2-compile2", "BUILD=2,COMPILE=2,RUN=4", 4),
            ):
                group = f"scheduling-{backend}-{label}"
                for state in ("cold", "warm"):
                    yield {
                        "name": f"{group}-{state}",
                        "group": group,
                        "seed_group": prime_group,
                        "suite": "scheduling",
                        "backend": backend,
                        "models": models,
                        "repeat": 0,
                        "per_stage": 1,
                        "parallel": 4,
                        "threads": threads,
                        "parallel_per_stage": limits,
                        "cache": state,
                    }
            for parallel in (1, 2):
                for label, ordered_models in (
                    ("grouped", ["sine_model", "sine_model", "aww", "aww"]),
                    ("interleaved", ["sine_model", "aww", "sine_model", "aww"]),
                ):
                    group = f"ordering-{backend}-p{parallel}-{label}"
                    yield {
                        "name": f"{group}-cold",
                        "group": group,
                        "suite": "ordering",
                        "backend": backend,
                        "models": ordered_models,
                        "repeat": 0,
                        "per_stage": 0,
                        "parallel": parallel,
                        "threads": 4,
                        "cache": "cold",
                    }
        if args.suite in ("all", "complexity"):
            for model in ("sine_model", "aww", "vww"):
                for repeat in range(args.repeats):
                    group = f"complexity-{backend}-{model}-r{repeat}"
                    states = ("off", "cold", "warm") if repeat % 2 == 0 else ("cold", "warm", "off")
                    for state in states:
                        yield {
                            "name": f"{group}-{state}",
                            "group": group,
                            "suite": "complexity",
                            "backend": backend,
                            "models": [model],
                            "repeat": repeat,
                            "per_stage": 0,
                            "parallel": 1,
                            "threads": 4,
                            "cache": state,
                        }
        if args.suite in ("all", "parallel"):
            # Four distinct models preserve batch size across scheduler/make settings.
            models = ["sine_model", "aww", "vww", "resnet"]
            prime_group = f"parallel-prime-{backend}"
            yield {
                "name": f"{prime_group}-cold",
                "group": prime_group,
                "suite": "parallel-prime",
                "backend": backend,
                "models": models,
                "repeat": 0,
                "per_stage": 0,
                "parallel": 4,
                "threads": 4,
                "cache": "cold",
            }
            for per_stage in (0, 1):
                for parallel, threads in ((1, 4), (2, 2), (4, 1), (4, 4)):
                    group = f"parallel-{backend}-s{per_stage}-p{parallel}-t{threads}"
                    states = ("off", "warm") if (parallel + per_stage) % 2 else ("warm", "off")
                    for state in states:
                        yield {
                            "name": f"{group}-{state}",
                            "group": group,
                            "seed_group": prime_group,
                            "suite": "parallel",
                            "backend": backend,
                            "models": models,
                            "repeat": 0,
                            "per_stage": per_stage,
                            "parallel": parallel,
                            "threads": threads,
                            "cache": state,
                        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "complexity", "parallel", "scheduling"), default="all")
    parser.add_argument("--backends", nargs="+", default=["tflmi", "tvmaotplus"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.home, args.raw, args.output = (path.resolve() for path in (args.home, args.raw, args.output))
    args.repo = Path(__file__).resolve().parents[1]
    args.workspace_vars = yaml.safe_load((args.home / "environment.yml").read_text()).get("vars", {})
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "ccache": subprocess.check_output(["ccache", "--version"], text=True).splitlines()[0],
        "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
        "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
        "home": str(args.home),
        "raw": str(args.raw),
    }
    # Keep the original machine record when resuming and record every invocation.
    if not (args.output / "machine.json").exists():
        (args.output / "machine.json").write_text(json.dumps(metadata, indent=2))
    metadata["started_utc"] = time.time()
    metadata["argv"] = sys.argv
    with (args.output / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps(metadata) + "\n")
    destination = args.output / "measurements.jsonl"
    completed = {}
    if destination.exists():
        completed = {item["name"]: item for item in map(json.loads, destination.read_text().splitlines())}
    seeds = {
        item["group"]: Path(item["destination"])
        for item in completed.values()
        if item["cache"] == "cold" and item["returncode"] == 0
    }
    count = 0
    for spec in plan(args):
        previous = completed.get(spec["name"])
        if previous is not None and previous["returncode"] == 0:
            continue
        if previous is not None:
            failed_dest = Path(previous["destination"])
            if failed_dest.exists():
                failed_dest.rename(failed_dest.with_name(failed_dest.name + f"-failed-{int(time.time())}"))
        seed = seeds.get(spec.get("seed_group", spec["group"]))
        result = sample(args, spec, seed=seed)
        with destination.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        if result["returncode"]:
            print(result["error"], flush=True)
            raise SystemExit(result["returncode"])
        if spec["cache"] == "cold":
            seeds[spec["group"]] = Path(result["destination"])
        count += 1
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()
