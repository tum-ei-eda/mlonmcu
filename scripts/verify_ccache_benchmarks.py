#!/usr/bin/env python3
"""Audit this benchmark matrix and independently recompile TVM's varying C unit."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile

from benchmark_ccache import plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Allow unfinished optional scheduling cases")
    args = parser.parse_args()
    directory = args.directory
    records = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]
    latest = {row["name"]: row for row in records}
    expected = []
    for suite in ("all", "scheduling"):
        expected.extend(plan(argparse.Namespace(suite=suite, backends=["tflmi", "tvmaotplus"], repeats=2)))
    missing = [spec["name"] for spec in expected if latest.get(spec["name"], {}).get("returncode") != 0]
    missing_primary = [name for name in missing if not name.startswith(("scheduling-", "ordering-"))]
    if missing_primary or (missing and not args.allow_partial):
        raise SystemExit(f"Incomplete matrix: {len(missing)} missing successful samples: {missing}")
    planned_count = len(expected)
    expected = [spec for spec in expected if spec["name"] not in missing]
    variants = defaultdict(set)
    resnet = {}
    model_runs = 0
    for spec in expected:
        row = latest[spec["name"]]
        for key, value in spec.items():
            assert row[key] == value, (spec["name"], key, row[key], value)
        assert len(row["runs"]) == len(row["models"])
        with (Path(row["destination"]) / "report.csv").open() as handle:
            report = list(csv.DictReader(handle))
        assert len(report) == len(row["models"])
        assert all(r.get("Failing", "False") != "True" for r in report)
        assert 0 < row["compile_union_s"] <= row["compile_sum_s"] + 1e-6
        assert row["compile_union_s"] <= row["wall_s"] + 1e-3
        for run in row["runs"]:
            model_runs += 1
            run_dir = Path(row["destination"]) / "runs" / str(run["run"])
            elf = run_dir / "mlif/bin/generic_mlonmcu"
            assert hashlib.sha256(elf.read_bytes()).hexdigest() == run["elf_sha256"], elf
            variants[row["backend"], run["model"]].add(run["elf_sha256"])
            for start, end in run["intervals"].values():
                assert math.isfinite(start) and math.isfinite(end) and end >= start
            if row["backend"] == "tvmaotplus" and run["model"] == "resnet" and row["cache"] != "off":
                resnet.setdefault(run["elf_sha256"], (row["name"], run_dir))
        if row["cache"] != "off":
            counters = row["cache_counters"]
            hits = counters["direct_cache_hit"] + counters["preprocessed_cache_hit"]
            assert hits == row["cache_hits"]
            assert counters["cache_miss"] == row["cache_misses"]
            assert counters.get("cleanups_performed", 0) == 0, row["name"]
            assert math.isclose(row["hit_rate"], hits / (hits + row["cache_misses"]))
    assert all(len(hashes) == 1 for (backend, _), hashes in variants.items() if backend == "tflmi")
    checks = []
    with tempfile.TemporaryDirectory(prefix="ccache-resnet-verify-") as temp:
        for index, (elf_hash, (name, run_dir)) in enumerate(resnet.items()):
            build = run_dir / "mlif"
            commands = json.loads((build / "compile_commands.json").read_text())
            entry = next(item for item in commands if item["file"].endswith("/default_lib1.c"))
            command = shlex.split(entry["command"])
            if Path(command[0]).name == "ccache":
                command = command[1:]
            cwd = Path(entry["directory"])
            original = cwd / command[command.index("-o") + 1]
            target = Path(temp) / f"{index}.o"
            for flag in ("-o", "-MF", "-MT"):
                if flag in command:
                    command[command.index(flag) + 1] = str(target.with_suffix(".d") if flag == "-MF" else target)
            subprocess.run(command, cwd=cwd, check=True, capture_output=True)
            original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            uncached_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            checks.append(
                {
                    "sample": name,
                    "elf_sha256": elf_hash,
                    "object": str(original),
                    "cached_object_sha256": original_hash,
                    "uncached_object_sha256": uncached_hash,
                    "match": original_hash == uncached_hash,
                    "command": command,
                }
            )
            assert checks[-1]["match"], name
    audit = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_samples": planned_count,
        "successful_samples": len(expected),
        "model_runs": model_runs,
        "missing": missing,
        "cache_evictions": 0,
        "elf_variants": {f"{backend}/{model}": sorted(hashes) for (backend, model), hashes in variants.items()},
        "resnet_uncached_recompilations": checks,
    }
    (directory / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    (directory / "resnet_cache_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Verified {len(expected)} samples, {model_runs} model runs and {len(checks)} resnet object variants")


if __name__ == "__main__":
    main()
