# MLIF ccache benchmark summary

The original 70-case benchmark matrix is complete. Including additional scheduling experiments, **85 successful samples covering 232 model runs** were collected and audited. Five optional cases remain unmeasured.

The study took longer than necessary because it was expanded with 20 extra scheduling tests and repeatedly ran full TVM generation, which ccache cannot accelerate. The extra suite was stopped after 15 completed samples.

See the [full report, tables and plots](README.md), [measurements CSV](measurements.csv), and [validation results](validation.json).

## Main results

Warm-cache speedups across `sine_model`, `aww`, and `vww`:

| Backend | Compile-stage speedup | End-to-end speedup |
|---|---:|---:|
| TFLMI | 29–30× | About 17× |
| TVMAOT+ | 1.07–1.28× | Only 1–4%; no reliable improvement established |

TFLMI reuses a large common runtime. TVMAOT+ spends most of its time generating model code, outside ccache’s scope.

These results describe `host_x86` builds with GCC 9.4.0 and ccache 3.7.7 on a shared 36-logical-CPU host. Single-model measurements have two repeats; parallelism and ordering comparisons are exploratory single samples. Warm means a populated cache with fresh run directories; cache seeding/copying is excluded from the timed warm flow.

Compile-stage time includes configuration, compilation, linking and artifact generation. The full report distinguishes accumulated compile time from the union of overlapping compile intervals and from end-to-end wall time.

## How ccache identifies the compiler

By default, ccache hashes the compiler’s **size and modification time**, alongside its name. Its absolute executable path alone does not establish identity.

- Replacing a compiler at the same path normally causes misses when its size or timestamp changes.
- Moving an identical compiler with the same name and metadata can preserve hits, provided the remaining toolchain and inputs are equivalent.
- The default check can treat changed bytes as the same compiler if both size and modification time are preserved.

To hash executable bytes, use:

```text
-c ccache.compiler_check=content
```

With this setting, timestamp-only changes can preserve hits, while changed executable bytes cause misses. Both behaviors were verified experimentally; see [compiler identity results](compiler_identity.json).

This hashes the driver executable, not the entire toolchain. Independently changed GCC `cc1` binaries or compiler libraries require additional file hashes through `extra_files_to_hash` in the managed `ccache.conf`, or a unique full-toolchain build ID through `compiler_check=string:<id>`. A version banner alone may not distinguish locally patched compilers.

Reference: [ccache 3.7.7 configuration documentation](https://ccache.dev/manual/3.7.7.html#_configuration_settings).

## Parallelism, cache warmth and run ordering

The useful TFLMI allocation reversed with cache state, despite the same **four-job compile budget**:

| Allocation | Initially empty cache | Populated cache |
|---|---:|---:|
| Four concurrent runs × one make job | 323.57 s | **7.16 s** |
| One run at a time × four make jobs | **98.81 s** | 12.10 s |

These stage-mode batches contain the same four models. Their BUILD worker limits differ, but TFLMI’s BUILD time is very short; the compile intervals show the same large difference.

Concurrent cold runs duplicated compilation: **403 misses versus 114**. Once warm, run concurrency improved throughput. Simultaneous equivalent misses can compile the same translation unit more than once.

Grouping identical model names was not automatically beneficial. Serial TFLMI batches took 90.72 seconds grouped and 90.89 seconds interleaved, with identical hit/miss totals. The two-worker ordering results were 90.08 and 95.05 seconds; that small single-sample difference should be treated cautiously. TVMAOT+ grouped/interleaved ordering was among the optional cases left unmeasured.

The flags control different resources:

- `--parallel` limits concurrent runs.
- `mlif.num_threads` controls make jobs within each MLIF compilation.
- `runs_per_stage=0` processes complete run pipelines concurrently.
- `runs_per_stage=1` introduces stage barriers and allows independent limits such as `--parallel-per-stage BUILD=4,COMPILE=1,RUN=4`.

For TVMAOT+, `BUILD=4,COMPILE=1` took about 120 seconds cold, compared with 243 seconds for `BUILD=1,COMPILE=4`. Parallel model generation mattered far more than parallel MLIF compilation.

Compile overlap also needs careful interpretation. With two TVMAOT+ workers, stage mode reduced the uncached compile union from 11.12 to 5.95 seconds, while end-to-end wall time changed only from 136.19 to 135.25 seconds. The compiler work was similar; its placement in time changed.

## Practical guidance

1. Warm common objects within each compatible compiler, flags and runtime family before launching many consumers when the reuse justifies the warm-up cost. This was valuable for TFLMI in this study.
2. Once warm, increase run concurrency while respecting CPU and memory budgets. Compare allocations at the same total make-job budget before attributing gains to scheduling.
3. Use `ccache.mode=workspace` when separate sessions should reuse the cache. The default session scope shares entries within that session.
4. For these TVMAOT+ workloads, prioritize BUILD concurrency. Compiler-cache hit rate alone is a poor predictor of full experiment speed.
5. Measure end-to-end batch wall time, compile overlap and CPU cost separately. More concurrency can reduce latency while increasing total CPU consumption.

`--parallel-per-stage` provides independent limits, but those limits remain static. There is no universal optimum independent of cache state, workload order, compiler inputs and available resources. Run indices are normalized by this feature; scheduling order matters through cache availability and load balancing during execution.
