#!/usr/bin/env python3
"""Export ccache measurements as CSV, Markdown tables and a comparison plot."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    records = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]
    latest = {row["name"]: row for row in records}
    samples = [row for row in latest.values() if row["returncode"] == 0]
    failures = [row for row in records if row["returncode"] != 0]
    fields = [
        "name",
        "suite",
        "backend",
        "cache",
        "repeat",
        "per_stage",
        "parallel",
        "threads",
        "parallel_per_stage",
        "wall_s",
        "build_union_s",
        "compile_sum_s",
        "compile_union_s",
        "compile_span_s",
        "cpu_user_s",
        "cpu_system_s",
        "cache_hits",
        "cache_misses",
        "hit_rate",
        "started_utc",
    ]
    with (directory / "measurements.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    groups = defaultdict(list)
    for row in samples:
        groups[(row["suite"], row["backend"], tuple(row["models"]), row["cache"])].append(row)

    def m(rows, key):
        return median(row[key] for row in rows)

    text = [
        "# MLIF ccache benchmarks: October 30 and November 2, 2026",
        f"{len(samples)} successful measurements; {len(failures)} failed attempts retained for provenance.",
        "## Measurement method",
        "Each sample starts a new CLI process and fresh run directories, completes LOAD, BUILD, COMPILE and RUN, "
        "and requires successful report rows. The target is host_x86. TVM/BLAS internal threads are held at one. "
        "The three complexity workloads are sine_model, aww and vww. Four-run parallel batches add resnet. "
        "These are default backend builds without autotuning; results do not generalize to every model or target.",
        "`off` disables ccache. `cold` starts an empty isolated session cache; later runs in that sample may share it. "
        "`warm` copies a populated cache before timing, then uses fresh directories. Cache copying is excluded from "
        "the timed CLI flow. Kernel page caches and dependency installations are not flushed: cold means ccache-cold, "
        "not machine-cold. Complexity samples have two repeats; parallelism and ordering are exploratory "
        "single samples.",
        "End-to-end wall time includes process startup, backend generation, MLIF configuration, compilation, linking, "
        "artifact export and execution. Compile **sum** accumulates each run's elapsed compile-stage interval and can "
        "exceed wall time; it is not CPU time. Compile **union** counts time with at least one active compile stage, "
        "without double-counting overlaps. Compile **span** also includes gaps between the first and last compile. "
        "COMPILE includes configuration, linking and artifact generation, not just compiler subprocesses. "
        "BUILD is backend model/code generation. All times below are seconds.",
        "The run resumed after Friday's interruption. Samples span two days on a shared 36-logical-CPU host, "
        "using ccache 3.7.7, GCC 9.4.0 and CMake 3.24.4. Load and child-process CPU time are recorded "
        "in the raw JSONL. "
        "Small timing differences and closely ranked configurations should not be treated as significant.",
        "## Increasing model complexity",
        "Medians across repeats. Speedup compares ccache off with warm ccache. "
        "Plot whiskers show the observed min–max range, not a confidence interval.",
    ]
    complexity = []
    plot_data = []
    for backend in ("tflmi", "tvmaotplus"):
        for model in ("sine_model", "aww", "vww"):
            states = [groups.get(("complexity", backend, (model,), state), []) for state in ("off", "cold", "warm")]
            if not all(states):
                continue
            off, cold, warm = states
            values = [m(rows, key) for key in ("compile_union_s", "wall_s") for rows in states]
            complexity.append(
                [
                    backend,
                    model,
                    "/".join(str(len(rows)) for rows in states),
                    *[f"{value:.2f}" for value in values],
                    f"{m(off, 'compile_union_s') / m(warm, 'compile_union_s'):.2f}x",
                    f"{m(off, 'wall_s') / m(warm, 'wall_s'):.2f}x",
                    f"{sum(r['cache_hits'] for r in warm)}/{sum(r['cache_misses'] for r in warm)}",
                ]
            )
            ranges = [
                (min(r[key] for r in rows), max(r[key] for r in rows))
                for key in ("compile_union_s", "wall_s")
                for rows in states
            ]
            plot_data.append((backend, model, values, ranges))
    text.append(
        table(
            [
                "Backend",
                "Model",
                "n off/cold/warm",
                "Compile off",
                "Compile cold",
                "Compile warm",
                "Wall off",
                "Wall cold",
                "Wall warm",
                "Compile speedup",
                "Wall speedup",
                "Warm total hits/misses",
            ],
            complexity,
        )
    )
    text.append("![Compile and end-to-end time by model](complexity.png)")
    text.append("## Four-run parallel batches")
    text.append(
        "Each batch contains sine_model, aww, vww and resnet in that order. P is `--parallel`; "
        "T is `mlif.num_threads`; S is `runs_per_stage`. P×T is a rough peak make-job budget, "
        "not a limit on backend threads or all process activity."
    )
    parallel = []
    for backend in ("tflmi", "tvmaotplus"):
        rows = [row for row in samples if row["suite"] == "parallel" and row["backend"] == backend]
        for s in (0, 1):
            for p, t in ((1, 4), (2, 2), (4, 1), (4, 4)):
                pair = {
                    row["cache"]: row
                    for row in rows
                    if (row["per_stage"], row["parallel"], row["threads"]) == (s, p, t)
                }
                if not all(state in pair for state in ("off", "warm")):
                    continue
                off, warm = pair["off"], pair["warm"]
                parallel.append(
                    [
                        backend,
                        s,
                        p,
                        t,
                        *[
                            f"{row[key]:.2f}"
                            for key in ("wall_s", "compile_sum_s", "compile_union_s")
                            for row in (off, warm)
                        ],
                        f"{off['wall_s'] / warm['wall_s']:.2f}x",
                        f"{warm['cache_hits']}/{warm['cache_misses']}",
                    ]
                )
    text.append(
        table(
            [
                "Backend",
                "S",
                "P",
                "T",
                "Wall off",
                "Wall warm",
                "Compile sum off",
                "Compile sum warm",
                "Compile union off",
                "Compile union warm",
                "Wall speedup",
                "Warm hits/misses",
            ],
            parallel,
        )
    )
    text.append("![Four-run parallelism comparisons](parallelism.png)")
    text.append("### CPU cost of the parallelism settings")
    text.append(
        "CPU is user plus system time for the timed CLI process and its waited-for descendants, "
        "in core-seconds. Average cores = CPU / wall time. These values help distinguish lower latency "
        "from lower total CPU cost; peak memory was not measured."
    )
    cpu_rows = []
    for backend in ("tflmi", "tvmaotplus"):
        for s in (0, 1):
            for p, t in ((1, 4), (2, 2), (4, 1), (4, 4)):
                pair = {
                    row["cache"]: row
                    for row in samples
                    if row["suite"] == "parallel"
                    and row["backend"] == backend
                    and (row["per_stage"], row["parallel"], row["threads"]) == (s, p, t)
                }
                if not all(state in pair for state in ("off", "warm")):
                    continue
                costs = [pair[state]["cpu_user_s"] + pair[state]["cpu_system_s"] for state in ("off", "warm")]
                cpu_rows.append(
                    [
                        backend,
                        s,
                        f"{p}/{t}",
                        *[f"{cost:.2f}" for cost in costs],
                        *[f"{cost / pair[state]['wall_s']:.2f}" for cost, state in zip(costs, ("off", "warm"))],
                    ]
                )
    text.append(
        table(["Backend", "S", "P/T", "CPU off", "CPU warm", "Average cores off", "Average cores warm"], cpu_rows)
    )
    text.extend(
        [
            "## Stage-specific worker limits and cold-cache scheduling",
            "S=1, P=4; T is listed per row. BUILD=1/COMPILE=4/T=1 and BUILD=4/COMPILE=1/T=4 have the same "
            "four-job compile budget, allocated differently. Each cold sample starts empty; "
            "each warm sample starts from the same backend's "
            "four-model seed. Unspecified stages use P=4. The initial fully concurrent cold seed is shown "
            "as a reference.",
        ]
    )
    scheduling = []
    for row in samples:
        if row["suite"] not in ("scheduling", "parallel-prime"):
            continue
        scheduling.append(
            [
                row["backend"],
                row.get("parallel_per_stage", "S=0, P=4"),
                row["threads"],
                row["cache"],
                f"{row['wall_s']:.2f}",
                f"{row['build_union_s']:.2f}",
                f"{row['compile_sum_s']:.2f}",
                f"{row['compile_union_s']:.2f}",
                f"{row['cpu_user_s'] + row['cpu_system_s']:.2f}",
                f"{row.get('cache_hits', 0)}/{row.get('cache_misses', 0)}",
            ]
        )
    text.append(
        table(
            [
                "Backend",
                "Limits",
                "T",
                "Cache",
                "Wall",
                "Build union",
                "Compile sum",
                "Compile union",
                "CPU",
                "Hits/misses",
            ],
            scheduling,
        )
    )
    text.extend(
        [
            "## Run ordering with an initially empty cache",
            "S=0, T=4. Grouped = [sine_model, sine_model, aww, aww]; interleaved = "
            "[sine_model, aww, sine_model, aww]. Same multiset of runs. Concurrency, shared library reuse and "
            "the point at which a reusable object becomes available can all affect time and hit count. "
            "A single ordering comparison is evidence about these batches, not a universal scheduling rule.",
        ]
    )
    ordering = []
    for row in samples:
        if row["suite"] == "ordering":
            ordering.append(
                [
                    row["backend"],
                    row["parallel"],
                    row["name"].split("-")[-2],
                    f"{row['wall_s']:.2f}",
                    f"{row['compile_sum_s']:.2f}",
                    f"{row['compile_union_s']:.2f}",
                    f"{row['cpu_user_s'] + row['cpu_system_s']:.2f}",
                    f"{row['cache_hits']}/{row['cache_misses']}",
                ]
            )
    text.append(
        table(["Backend", "P", "Order", "Wall", "Compile sum", "Compile union", "CPU", "Hits/misses"], ordering)
    )
    text.extend(
        [
            "## Compiler identity",
            "An isolated experiment copies GCC to two different absolute paths with the same basename. "
            "It then changes only mtime, then appends inert bytes to the ELF executable. "
            "`compiler_identity.json` contains cumulative counters; the table shows increments for each invocation.",
        ]
    )
    identity = []
    prior = {}
    for row in json.loads((directory / "compiler_identity.json").read_text()):
        counters = row["counters"]
        current = (counters["direct_cache_hit"] + counters["preprocessed_cache_hit"], counters["cache_miss"])
        before = prior.get(row["check"], (0, 0))
        identity.append([row["check"], row["case"], current[0] - before[0], current[1] - before[1]])
        prior[row["check"]] = current
    text.append(table(["Check", "Case", "Hits", "Misses"], identity))
    text.append(
        "The default is `mtime`: compiler size and modification time contribute to the cache key, "
        "along with the compiler name. Absolute path alone does not establish identity. "
        "`-c ccache.compiler_check=content` hashes executable bytes, avoiding misses from timestamp-only "
        "changes and detecting different bytes at the same path. This does not fingerprint an entire toolchain: "
        "separate GCC cc1 binaries, plugins or dynamic libraries can change independently. For those builds, "
        "use an immutable full-toolchain build ID (`string:<id>`) or `extra_files_to_hash` in ccache.conf. "
        "A shared version banner may hide local patches. "
        "[ccache 3.7.7 manual](https://ccache.dev/manual/3.7.7.html#_configuration_settings)"
    )
    text.extend(
        [
            "## Interpretation and scheduling guidance",
            "TFLMI compiles a large common runtime, so model size alone need not increase MLIF compile time. "
            "Sharing those objects is valuable even between different models. TVMAOT+ generates model-specific C "
            "in BUILD; ccache only accelerates the later MLIF compilation. Its end-to-end benefit depends on how "
            "much of total time is spent in that stage and whether regenerated sources match.",
            "Fresh TVM invocations can emit equivalent C in different declaration orders. The preserved "
            "`tvm_sine_regeneration.diff` shows this for the first sine_model cold/warm pair. That pair had "
            "17 hits and one miss; all three initial sine_model binaries were byte-identical. A changed generated "
            "timestamp comment alone can fall back to a preprocessed hit, but reordered C can still require "
            "compilation. Repeating the same model therefore does not guarantee a fully warm compile.",
            "Resnet also produced multiple ELF variants when pointer declarations changed order. "
            "The preserved `tvm_resnet_default_lib1.diff` records the generated source difference. "
            "The final audit independently recompiles that C unit for each recorded cache-enabled resnet "
            "ELF variant and compares its object bytes. See `resnet_cache_verification.json`.",
            "Optimize batch wall time under a CPU and memory budget. For empty caches, completing a representative "
            "run from each genuinely compatible compiler/flags/runtime family before launching many consumers can "
            "reduce duplicate misses. That trades some initial concurrency for reuse; incompatible flags or compiler "
            "builds do not share those entries. ccache is not a dependency scheduler and should not be assumed to "
            "deduplicate simultaneous misses into a single compilation.",
            "For mostly warm caches, more concurrent runs and fewer make threads per run can improve throughput "
            "under a fixed compile-job budget; "
            "configuration, linking and filesystem traffic eventually limit scaling. For CPU-heavy backend generation, "
            "use stage limits to control BUILD independently of COMPILE. Stage mode also introduces barriers: "
            "BUILD=1 serializes every model's backend generation before any MLIF compilation starts.",
            "Measure both first-use and steady-state workloads, keep the run multiset and cache seeds fixed, "
            "and repeat promising configurations with alternating order. Tune separate cold and warm policies "
            "rather than seeking one universally optimal P/T setting. Session mode starts empty for each new session; "
            "workspace mode is useful when later experiments should reuse earlier results.",
            "## Artifacts and reproduction",
            "`measurements.jsonl` is append-only and includes failed attempts; `measurements.csv` contains the latest "
            "successful sample for each name. Per-run intervals, ELF SHA-256 digests, commands, cache counters, "
            "load and CPU usage are retained in JSONL. Raw commands and reports are under "
            "`/tmp/mlonmcu-ccache-bench-20261030`. `machine.json` records the original machine; "
            "`invocations.jsonl` records subsequent runner invocations. Incomplete interrupted directories "
            "are preserved.",
            r"""```bash
venv/bin/python scripts/benchmark_ccache.py --home workspace_default \
  --raw /tmp/mlonmcu-ccache-bench-20261030 \
  --output benchmarks/ccache_2026-10-30 --suite all --repeats 2
venv/bin/python scripts/benchmark_ccache.py --home workspace_default \
  --raw /tmp/mlonmcu-ccache-bench-20261030 \
  --output benchmarks/ccache_2026-10-30 --suite scheduling
venv/bin/python scripts/benchmark_ccache_compiler_identity.py \
  --output benchmarks/ccache_2026-10-30/compiler_identity.json
venv/bin/python scripts/verify_ccache_benchmarks.py benchmarks/ccache_2026-10-30 --allow-partial
venv/bin/python scripts/summarize_ccache_benchmarks.py benchmarks/ccache_2026-10-30
```""",
            "Generated at " + datetime.now(timezone.utc).isoformat(),
        ]
    )
    findings = ["## Findings"]
    for backend in ("tflmi", "tvmaotplus"):
        speedups = []
        for model in ("sine_model", "aww", "vww"):
            off = groups.get(("complexity", backend, (model,), "off"), [])
            warm = groups.get(("complexity", backend, (model,), "warm"), [])
            if off and warm:
                speedups.append(
                    (m(off, "compile_union_s") / m(warm, "compile_union_s"), m(off, "wall_s") / m(warm, "wall_s"))
                )
        if speedups:
            findings.append(
                f"**{backend}:** warm-cache compile speedup across the three models was "
                f"{min(s[0] for s in speedups):.2f}–{max(s[0] for s in speedups):.2f}x; "
                f"end-to-end speedup was {min(s[1] for s in speedups):.2f}–"
                f"{max(s[1] for s in speedups):.2f}x. "
                + (
                    "The TVM end-to-end differences are small relative to normal timing variation; "
                    "two repeats do not establish a reliable small improvement."
                    if backend == "tvmaotplus"
                    else "Most of the uncached compilation is the common TFLM runtime."
                )
            )
    first = "scheduling-tflmi-build1-compile4"
    second = "scheduling-tflmi-build4-compile1"
    needed = [f"{group}-{state}" for group in (first, second) for state in ("cold", "warm")]
    if all(name in latest and latest[name]["returncode"] == 0 for name in needed):
        a, b = latest[f"{first}-cold"], latest[f"{second}-cold"]
        aw, bw = latest[f"{first}-warm"], latest[f"{second}-warm"]
        findings.append(
            "**Cache state changes the useful allocation of a four-job compile budget.** "
            f"For TFLMI, four cold runs with one make job each took {a['wall_s']:.2f}s "
            f"({a['cache_misses']} misses), versus {b['wall_s']:.2f}s ({b['cache_misses']} misses) "
            "for one run at a time with four make jobs. With populated caches those allocations took "
            f"{aw['wall_s']:.2f}s and {bw['wall_s']:.2f}s, respectively. "
            "These stage-mode comparisons have different BUILD limits, but TFLMI's BUILD time is very short; "
            "the compile interval and CPU measurements in the tables expose the difference directly."
        )
    findings.append(
        "**Scheduling guidance:** budget backend workers and compiler jobs separately. Empty shared caches can "
        "benefit from warming common runtime objects before launching consumers; warm caches often benefit from "
        "more concurrent runs. Group only compatible compiler/flags/runtime families. Run indices are normalized "
        "by this feature; it is cache availability during execution that makes scheduling order significant. "
        "The measured parallelism settings are single samples, not a proven universal optimum."
    )
    serial_stages = latest.get("parallel-tvmaotplus-s0-p2-t2-off")
    barriers = latest.get("parallel-tvmaotplus-s1-p2-t2-off")
    if serial_stages and barriers and serial_stages["returncode"] == barriers["returncode"] == 0:
        findings.append(
            "**Compile overlap is not the same as end-to-end speedup.** For TVMAOT+ P=2/T=2, stage mode "
            f"changed compile union from {serial_stages['compile_union_s']:.2f}s to "
            f"{barriers['compile_union_s']:.2f}s, while wall time changed only from "
            f"{serial_stages['wall_s']:.2f}s to {barriers['wall_s']:.2f}s. "
            "The compile sums stayed similar: the stage barrier aligned the compilations in time."
        )
    if (directory / "validation.json").exists():
        audit = json.loads((directory / "validation.json").read_text())
        findings.append(
            f"The final audit verified {audit['successful_samples']} samples and {audit['model_runs']} model runs, "
            "their report rows, saved ELF hashes, cache accounting and independently recompiled "
            "resnet object variants. "
            "See `validation.json`. Source revisions and local dependency changes are listed in `dependencies.json`."
        )
        if audit["missing"]:
            findings.append(
                "The original 70-case matrix is complete. The extra scheduling study was stopped after "
                f"{audit['successful_samples'] - 70} completed optional samples; "
                f"{len(audit['missing'])} optional cases remain unmeasured. "
                "The interrupted TVM warm sample is excluded. TVM grouped/interleaved ordering was not measured. "
                "The missing names are recorded in `validation.json`."
            )
    text[2:2] = findings
    (directory / "README.md").write_text("\n\n".join(text) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV and report generated")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, offset, title in zip(axes, (0, 3), ("MLIF compile stage", "End-to-end CLI flow")):
        for state_index, label in enumerate(("Off", "Cold", "Warm")):
            ax.bar(
                [i + (state_index - 1) * 0.25 for i in range(len(plot_data))],
                [row[2][offset + state_index] for row in plot_data],
                width=0.25,
                label=label,
                yerr=[
                    [row[2][offset + state_index] - row[3][offset + state_index][0] for row in plot_data],
                    [row[3][offset + state_index][1] - row[2][offset + state_index] for row in plot_data],
                ],
                capsize=2,
            )
        ax.set_xticks(range(len(plot_data)), [f"{row[0]}\n{row[1]}" for row in plot_data], rotation=45, ha="right")
        ax.set_ylabel("Seconds (median)")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "complexity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row_index, backend in enumerate(("tflmi", "tvmaotplus")):
        configurations = [(s, p, t) for s in (0, 1) for p, t in ((1, 4), (2, 2), (4, 1), (4, 4))]
        for column, key in enumerate(("wall_s", "compile_union_s")):
            ax = axes[row_index, column]
            for state_index, state in enumerate(("off", "warm")):
                measured = {
                    (r["per_stage"], r["parallel"], r["threads"]): r[key]
                    for r in samples
                    if r["suite"] == "parallel" and r["backend"] == backend and r["cache"] == state
                }
                ax.bar(
                    [i + (state_index - 0.5) * 0.35 for i in range(len(configurations))],
                    [measured.get(config, float("nan")) for config in configurations],
                    width=0.35,
                    label=state,
                )
            ax.set_xticks(range(len(configurations)), [f"S{s}\nP{p}/T{t}" for s, p, t in configurations])
            if any(r["suite"] == "parallel" and r["backend"] == backend for r in samples):
                ax.set_yscale("log")
            ax.set_ylabel("Seconds (log scale)")
            ax.set_title(f"{backend}: {'end-to-end' if column == 0 else 'compile union'}")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "parallelism.png", dpi=160)
    plt.close(fig)
    print(f"Wrote report, CSV and plot for {len(samples)} successful samples")


if __name__ == "__main__":
    main()
