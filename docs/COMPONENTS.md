# Support

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
- ARM Model Zoo ([`ARM-software/ML-zoo`](https://github.com/ARM-software/ML-zoo)) :warning: Work in Progress :warning:

## Frameworks and Backends

For each framework supported by MLonMCU, a number of backends is implemented.

| Framework     | Backends                                                                        |
|---------------|---------------------------------------------------------------------------------|
| TenflowFlow Lite<br>for Microcontrollers (`tflm`) | Default Interpreter (`tflmi`)<br>Offline Compiler (`tflmc`) :warning: Work in Progress :warning:        |
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

## Features

Table

| Feature                                             | Setup | Frontend | Framework | Backend | Target | Platform/Compile | Other |
|-----------------------------------------------------|-------|----------|-----------|---------|--------|------------------|-------|
| Debug Arena Usage (`debug_arena`)                   | x     |          |           |         |        |                  |       |
| Validate Output Data (`validate`)                   | x     | x        |           |         |        |                  |       |
| muRISCV-NN (`muriscvnn`)                            | x     | x        |           |         |        |                  |       |
| CMSIS-NN (`cmsisnn`)                                | x     | x        |           |         |        |                  |       |
| CMSIS-NN + TVM BYOC (`cmsisnnbyoc`)                 | x     |          |           |         |        |                  |       |
| Fused TIling for TVM (`fusetile`)                   | x     |          |           |         |        |                  |       |
| Custom TVM meory planner (`memplan`)                | x     |          |           |         |        |                  |       |
| Unified Static Memory Planner for TVM (`usmp`) TODO | x     |          |           |         |        |                  |       |
| V-Extension for RISC-V (`vext`)                     | x     | x        |           |         |        |                  |       |
| Debug Build (`debug`)                               | x     | x        |           |         |        |                  |       |
| GDBServer (`gdbserver`)                             | x     |          |           |         |        |                  |       |
| Debug ETISS VP (`etissdbg`)                         | x     |          |           |         |        |                  |       |
| Create Memory Trace (`trace`)                       | x     |          |           |         |        |                  |       |
| Unpacked API (`unpacked_api`)                       | x     |          |           |         |        |                  |       |
| Autotune TVM Model (`autotune`)                     | x     |          |           |         |        |                  |       |
| Use Tuning ? (`autotuned`)                          | x     |          |           |         |        |                  |       |


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
