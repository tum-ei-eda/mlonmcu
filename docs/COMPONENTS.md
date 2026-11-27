# Components

## Models and Frontends

The types of models which can be processed with MLonMCU are given by the implemented frontends. The Following Table shows the currently supported ones:

| Frontend | Formats   |
|----------|-----------|
| TFLite   | `.tflite` |

Here is also a list of frontends with will probably implemented in the furture:

- ONNX (`onnx`)
- TensorFlow SavedModel (`.pb`)

While you can use your own models we also provide support for the following model zoos which can be cloned from GitHub:

- Model Collection by EDA@TUM ([`tum-ei-eda/mlonmcu-models`](https://github.com/tum-ei-eda/mlonmcu-models))
- ARM Model Zoo ([`ARM-software/ML-zoo`](https://github.com/ARM-software/ML-zoo)) ⚠️ Work in Progress ⚠️

## Frameworks and Backends

For each framework supported by MLonMCU, a number of backends is implemented.

| Framework     | Backends                                                                        |
|---------------|---------------------------------------------------------------------------------|
| TenflowFlow Lite<br>for Microcontrollers (`tflm`) | Default Interpreter (`tflmi`)<br>Offline Compiler (`tflmc`) ⚠️ Work in Progress ⚠️        |
| TVM (`tvm`)   | AoT Executor (`tvmaot`)<br>Graph Executor (`tvmrt`)<br>Custom Codegenerator (`tvmcg`) |

## Platforms and Targets

While support for some targets (especially simulator based ones) is directly build into MLonMCU, a platform is used for more complivcated targets (e.g. real hardware) to reuse existing Flows for compiling and flashing

| Platform           | Targets                                                                                                                                                                                   |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Default            | **RISC-V:**<br>ETISS/Pulpino (`etiss_pulpino`)<br>Spike (`spike`)<br>VPSimPlus (`ovpsim`)<br><br>**ARM:**<br>Corstone300 FVP (`corstone300`)<br><br>**x86:**<br>Host `host_x86`) |
| ESP IDF (`espidf`) | **LX6:**<br>ESP32 (`esp32`)<br><br>**RISC-V:**<br>ESP32-C3 (`esp32c3`)                                                                                                                        |

To extend the support of read hardware targets, it would be great if this list would be extended by some of the following platforms in the furture:

- Arduino Excosystem
- PlatformIO
- ZephyrOS

### Scalar instruction benchmarks on ESP32-P4

The `rvv_bench` frontend supports `instructions_scalar` on ESP-IDF, reusing the
benchmark sources in the `mlonmcu-sw` checkout configured by `mlif.src_dir`.
The ESP-IDF project template is selected automatically. No ML framework or
backend is needed. Standard RVV instruction benchmarks are not supported on ESP32-P4.

```bash
python -m mlonmcu.cli.main flow run instructions_scalar \
  --platform espidf -t esp32p4 --post rvv_bench \
  -c espidf.src_dir=/work/esp-idf-v6.0 \
     espidf.install_dir=/work/espressif-v6.0 \
     esp32p4.revision=1.3 \
     espidf.port=/dev/ttyACM0 \
     espidf.wait_for_user=false \
     espidf.use_idf_monitor=false
```

The workspace dependency cache normally supplies `mlif.src_dir`; it can also be
passed with `-c mlif.src_dir=/path/to/mlonmcu-sw`. The port can be a stable
`/dev/serial/by-id/...` path. The revision setting must match the board.
This command flashes the board, captures UART output in `esp32p4_out.log`, and
generates `rvv_bench.html` in the run directory. The same postprocess works with
instruction benchmark output from MLIF targets such as ETISS.

Measurements use the upstream warmup/unroll/loop/run settings and the CPU cycle
counter. They include loop overhead and effects from caches, alignment and
FreeRTOS interrupts; treat them as repeated-loop timings, rather than isolated
instruction latency. The upstream memory tests use offset 13, so halfword and
word accesses are unaligned.

### PIE v1 instruction POC on ESP32-P4

Use `instructions_pie` in the command above to benchmark a subset of the vendor
SIMD instructions on pre-v3 hardware (`Xespv2p1`). The build rejects the newer
PIE ISA configuration; pass `esp32p4.revision=1.3` for a revision-1.3 board.

The suite contains 32 cases, grouped into calibration, independent-input
instructions, dependent instruction chains, accumulator/fused-memory operations,
and producer/consumer sequences. It covers vector add/subtract/min/max,
AND/OR/XOR/NOT/zero, 128-bit load/store, signed 8-/16-bit multiply, 32-bit shifts,
QACC/XACC multiply-accumulate, fused MAC/load, and QR/AR transfers.
The sources are bundled with the ESP-IDF project template. Every case executes
once against a scalar reference before timing, including saturation boundaries,
QACC conversion truncation, and signed 40-bit XACC readout.

Timing loops run from IRAM with aligned internal DRAM buffers. Memory tests use
zero post-increment to remain in bounds. Each sample contains 512 loops of 64 test
iterations after warmup; results are medians of five samples. The report includes
an empty-loop/counter baseline. Single-instruction rows measure cycles per
instruction; explicitly labeled two-instruction rows measure cycles per entire
sequence, without dividing by two. Dependency-chain rows feed each result into
the next instruction; independent-input rows characterize repeated-loop throughput.
Accumulator rows have implicit dependencies. Arithmetic data changes during
chains and accumulation, so these tests do not characterize every operand pattern.
All timings include loop/counter overhead and interrupts; memory cases access hot
buffers and do not measure streaming bandwidth. The HTML report preserves the
groups, units, and measurement settings. An optional
`-c rvv_bench.title="ESP32-P4 PIE v1 instruction timings"` customizes its title.

## Features

An extensive overview of the available features in TVM is given in the following table. The types of those features are denoted with a check mark in the respective column.

| Feature                                             | Setup              | Frontend           | Framework          | Backend            | Target             | Platform/Compile   | Other |
|-----------------------------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-------------------------|-------|
| Debug Arena Usage (`debug_arena`)                   |                    |                    |                    |                    |                    | ✅ |       |
| Validate Output Data (`validate`)                   |                    | ✅ |                    |                    |                    | ✅ |       |
| muRISCV-NN (`muriscvnn`)                            | ✅ |                    | ✅ |                    |                    |                                    |       |
| CMSIS-NN (`cmsisnn`)                                | ✅ |                    | ✅ |                    |                    |                                    |       |
| CMSIS-NN + TVM BYOC (`cmsisnnbyoc`)                 |                    |                    |                    | ✅ |                    |                                    |       |
| Fused TIling for TVM (`fusetile`)                   |                    |                    |                    | ✅ |                    |                                    |       |
| Custom TVM meory planner (`memplan`)                |                    |                    |                    | ✅ |                    |                                    |       |
| Unified Static Memory Planner for TVM (`usmp`)      |                    |                    |                    | ✅ |                    |                                    |       |
| V-Extension for RISC-V (`vext`)                     | ✅ |                    |                    |                    | ✅<br>(`spike`, `ovpsim`) |                                    |       |
| Debug Build (`debug`)                               | ✅ |                    |                    |                    |                    |                  ✅ |       |
| GDBServer (`gdbserver`)                             |                    |                    |                    |                    | ✅ |                                    |       |
| Debug ETISS VP (`etissdbg`)                         |                    |                    |                    |                    | ✅<br>(`etiss_pulpino`) |                                    |       |
| Create Memory Trace (`trace`)                       |                    |                    |                    |                    | ✅<br>(`etiss_pulpino`) |                                    |       |
| Unpacked API (`unpacked_api`)                       |                    |                    |                    | ✅<br>(`tvmaot`) |                    |                                    |       |
| Autotune TVM Model (`autotune`)                     |                    |                    |                    | ✅ |                    |                                    |       |
| Use TVM Tuning Records (`autotuned`)                |                    |                    |                    | ✅ |                    |                                   |       |


## Managed Dependencies

MLonMCU tries to either manage dependencies internally (hidden to the user) or rely on 3rd party platforms to install them.

The following list gives and overview of the set of dependencies which are currently managed:

- Toolchains
	- RISC-V GCC Linux Toolchain
		- Download and extract
	- ARM GCC Linux Toolchain
		- Download and extract
	- LLVM
		- Download and extract
		
- Targets/Simulators
	- ETISS
		- Clone Repository
		- Build ETISS
		- Install ETISS
		- Build `bare_etiss_processor`
	- Spike
		- Clone Repositories
		- Build Proxy Kernel
		- Build Simulator
	- Corstone-300
		- Download and extract
		- Install FVP

- Frameworks/Backends
	- TFLM
		- Clone Repository
		- Download 3rd party dependencies
	- TFLite Micro Compiler
		- Clone Repository
		- Build
	- TVM
		- Clone Repository (including 3rd party dependencies)
		- Configure & Build
	- uTVM Staticrt Codegen
		- Clone Repository
		- Build
		
- Features
	- muRISCV-NN
		- Clone Repository
		- Build
	- CMSIS(-NN)
		- Clone Repository
		- Build

The following dependencies are intentionally NOT managed by MLonMCU:

- OVPSimPlus: The simulator is closed source and needs an individual license for usage (free)
- ESP-IDF: Make sure you provide a `espidf.path` with the required components installed
  - For ESP32-P4, optionally pass `-c esp32p4.revision=1.3` to select the hardware revision.
    Revisions `0.x` and `1.x` enable `CONFIG_ESP32P4_SELECTS_REV_LESS_V3`; revisions `3.x` disable it.
    MLonMCU selects the highest ESP-IDF minimum revision that includes the requested hardware
    (for example, v1.0 for a v1.3 chip). Omit the option to retain ESP-IDF's defaults.
    Start a new run after changing the revision to regenerate the project and bootloader.
