#!/usr/bin/env python3
"""Check installed ccache's compiler identity behavior using isolated compiler copies."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="ccache-compiler-identity-") as temp:
        root = Path(temp)
        compiler = Path(shutil.which("gcc")).resolve()
        copies = []
        for directory in ("a", "b"):
            target = root / directory / "gcc"
            target.parent.mkdir()
            shutil.copy2(compiler, target)
            copies.append(target)
        source = root / "source.c"
        source.write_text("int value(void) { return 42; }\n")
        # Keep source timestamps safely older than ccache's invocation time.
        os.utime(source, (1_600_000_000, 1_600_000_000))
        cc1 = subprocess.check_output([str(compiler), "-print-prog-name=cc1"], text=True).strip()
        env = {key: value for key, value in os.environ.items() if not key.startswith("CCACHE_")}
        for check in ("mtime", "content"):
            cache = root / f"cache-{check}"
            cache.mkdir()
            config = cache / "ccache.conf"
            config.touch()
            env.update(CCACHE_DIR=str(cache), CCACHE_CONFIGPATH=str(config), CCACHE_COMPILERCHECK=check)

            def compile_one(label, executable):
                subprocess.run(
                    [
                        "ccache",
                        str(executable),
                        "-B" + str(Path(cc1).parent) + "/",
                        "-c",
                        str(source),
                        "-o",
                        str(root / "output.o"),
                    ],
                    env=env,
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                counters = dict(
                    (key, int(value))
                    for key, value in (
                        line.split()
                        for line in subprocess.check_output(
                            ["ccache", "--print-stats"], env=env, text=True
                        ).splitlines()
                    )
                )
                results.append({"check": check, "case": label, "counters": counters})

            for target in copies:
                shutil.copy2(compiler, target)
            compile_one("initial", copies[0])
            compile_one("different_absolute_path_same_name_and_metadata", copies[1])
            stat = copies[1].stat()
            os.utime(copies[1], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
            compile_one("identical_bytes_changed_mtime", copies[1])
            # ELF tolerates trailing bytes; simulate a changed compiler executable
            # without needing to rebuild a compiler or changing generated code.
            with copies[1].open("ab") as handle:
                handle.write(b"ccache-identity-experiment")
            compile_one("changed_bytes", copies[1])
            stat = copies[1].stat()
            with copies[1].open("r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b"X")
            os.utime(copies[1], ns=(stat.st_atime_ns, stat.st_mtime_ns))
            compile_one("changed_bytes_same_size_and_mtime", copies[1])
            renamed = root / "b" / "other-gcc"
            shutil.copy2(compiler, renamed)
            compile_one("identical_bytes_different_basename", renamed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        counters = row["counters"]
        print(
            row["check"],
            row["case"],
            "hits=",
            counters["direct_cache_hit"] + counters["preprocessed_cache_hit"],
            "misses=",
            counters["cache_miss"],
        )


if __name__ == "__main__":
    main()
