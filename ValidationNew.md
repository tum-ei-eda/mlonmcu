# MLonMCU - Validation (new)

## Updates

- `mlonmcu-models` (`$MLONMCU_HOME/models/`, Branch: `refactor-validate`):
	- `resnet/definition.yml`
		- Add input/output dtype
		- Add dequantize details (taken from `mlif_override.cpp`)
	- `resnet/support/mlif_override.cpp`
		- Comment out old validation code
		- Add example code to dump raw inputs/outputs via stdin/stdout
- `mlonmcu-sw` (`$MLONMCU_HOME/deps/src/mlif`):
	- Nothing changed (yet)
- `mlonmcu` (Branch: `refactor-validate`):
	- `mlonmcu/target/common.py`
		- Allow overriding used encoding for stdout (Required for dumping raw data)
		- Add optional `stdin_data` argument for passing input to process (Only works with `live=False` aka. `-c {target}.print_outputs=0`)
	- `mlonmcu/target/target.py` / `mlonmcu/target/riscv/spike.py`
		- `target.exec()` may return artifacts now
		- Add checks: `supports_filesystem`, `supports_stdout`, `supports_stdin`, `supports_argv`, `supports_uart`
	- `mlonmcu/feature/feature.py`:
		- Add `gen_data` feature
		- Add `gen_ref_data` feature
		- Add `set_inputs` feature
		- Add `get_outputs` feature
		- Add wrapper `validate_new` feature (Limitation: can not enable postprocess yet)
	- `mlonmcu/models/frontend.py`
		- Parse input/output types from `definition.yml`
		- Parse quantization/dequantization details from `definition.yml`
		- Create `model_info.yml` Artifact to pass input/output dtypes/shapes/quant to later stages
		- Implement `gen_data` functionality (including workaround to convert raw `inputs/0.bin` to numpy)
		- Implement dummy `gen_ref_data` functionality
	- `mlonmcu/platform/mlif/mlif.py` / `mlonmcu/platform/mlif/mlif_target.py`
		- Implement `set_inputs` functionality (stdin_raw only)
		- Implement `get_outputs` functionality (stdout_raw only)
		- Implement batching (run simulation serveral {num_inputs/batch_size} times, with different inputs)
	- `mlonmcu/platform/tvm/tvm_target_platform.py` / `mlonmcu/platform/tvm/tvm_target.py`
		- Implement `set_inputs` functionality (filesystem only)
		- Implement `get_outputs` functionality (filesystem only)
		- Implement batching (run simulation serveral {num_inputs/batch_size} times, with different inputs)
	- `mlonmcu/session/postprocess/postprocesses.py`
		- Add `ValidateOutputs` postprocess (WIP)




## Examples

```
# platform: tvm target: tvm_cpu`
# implemented:
python3 -m mlonmcu.cli.main flow run resnet -v \
	--target tvm_cpu --backend tvmllvm \
	--feature validate_new --post validate_outputs \
	-c tvm.print_outputs=1 -c tvm_cpu.print_outputs=1 \
	-c set_inputs.interface=filesystem -c get_outputs.interface=filesystem
python3 -m mlonmcu.cli.main flow run resnet -v \
	--target tvm_cpu --backend tvmllvm \
	--feature validate_new --post validate_outputs \
	-c tvm.print_outputs=1 -c tvm_cpu.print_outputs=1 \
	-c set_inputs.interface=auto -c get_outputs.interface=auto

# not implemented:
# python3 -m mlonmcu.cli.main flow run resnet -v \
#	  --target tvm_cpu --backend tvmllvm \
#	  --feature validate_new --post validate_outputs \
#	  -c tvm.print_outputs=1 -c tvm_cpu.print_outputs=1
#	  -c set_inputs.interface=filesystem -c get_outputs.interface=stdout

# platform: mlif target: spike
# implemented:
python3 -m mlonmcu.cli.main flow run resnet -v \
--target spike  --backend tvmaotplus    \
-f validate \
--feature validate_new --post validate_outputs \
-c mlif.print_outputs=1 -c spike.print_outputs=0 \
-c set_inputs.interface=stdin_raw -c get_outputs.interface=stdout_raw

# not implemented:
# python3 -m mlonmcu.cli.main flow run resnet -v \
#   --target spike  --backend tvmaotplus    \
#   -f validate \
#   --feature validate_new --post validate_outputs \
#   -c mlif.print_outputs=1 -c spike.print_outputs=1 \
#   -c set_inputs.interface=stdin -c get_outputs.interface=stdout
# python3 -m mlonmcu.cli.main flow run resnet -v \
#   --target spike  --backend tvmaotplus    \
#   -f validate \
#   --feature validate_new --post validate_outputs \
#   -c mlif.print_outputs=1 -c spike.print_outputs=1 \
#   -c set_inputs.interface=filesystem -c get_outputs.interface=filesystem
# python3 -m mlonmcu.cli.main flow run resnet -v \
#   --target spike  --backend tvmaotplus    \
#   -f validate \
#   --feature validate_new --post validate_outputs \
#   -c mlif.print_outputs=1 -c spike.print_outputs=0 \
#   -c set_inputs.interface=auto -c get_outputs.interface=auto
# combinations (e.g. filesystem+stdout) should also be tested!

# platform: mlif target: host_x86`
# TODO: should support same configs as spike target
```

## TODOs
- [ ] Fix broken targets (due to refactoring of `self.exec`) -> PHILIPP
- [ ] Add missing target checks (see above) -> PHILIPP
- [ ] Update `definition.yml` for other models in `mlonmcu-sw` (At least `aww`, `vww`, `toycar`) -> LIU
- [ ] Refactor model support (see `mlomcu_sw/lib/ml_interface`) to be aware of output/input tensor index (maybe even name?) und sample index -> PHILIPP
- [ ] Write generator for custom `mlif_override.cpp` (based on `model_info.yml` + `in_interface` + `out_interface` (+ `inputs.npy`)) -> LIU
- [ ] Eliminate hacks used to get `model_info.yml` and `inputs.yml` in RUN stage -> PHILIPP
- [ ] Implement missing interfaces for tvm (out: `stdout`) -> LIU
- [ ] Implement missing interfaces for mlif platform (in: `filesystem`, `stdin`; out: `filesystem`, `stdout`) -> LIU
- [ ] Implement missing interfaces for mlif platform (in: `rom`) -> PHILIPP)
- [ ] Add support for multi-output/multi-input -> PHILIPP/LIU
- [ ] Update `gen_data` & `gen_ref_data` feature (see NotImplementedErrors, respect fmt,...) -> LIU
- [ ] Move `gen_data` & `gen_ref_data` from LOAD stage to custom stage (remove dependency on tflite frontend) -> PHILIPP
- [ ] Test with targets: `tvm_cpu`, `host_x86`, `spike` (See example commands above) -> LIU
- [ ] Extend `validate_outputs` postprocess (Add `report`, implement `atol`/`rtol`, `fail_on_error`, `top-k`,...) -> LIU
- [ ] Add more validation data (at least 10 samples per model, either manually or using `gen_ref_outputs`) -> LIU
- [ ] Generate validation reports for a few models (`aww`, `vww`, `resnet`, `toycar`) and at least backends (`tvmaotplus`, `tflmi`) -> LIU
- [ ] Cleanup codebases (lint, remove prints, reuse code, create helper functions,...) -> LIU/PHILIPP
- [ ] Document usage of new validation feature -> LIU
- [ ] Add tests -> LIU/PHILIPP
- [ ] Streamline `model_info.yml` with BUILD stage `ModelInfo` -> PHILIPP
- [ ] Improve artifacts handling -> PHILIPP
- [ ] Support automatic quantization of inputs (See `vww` and `toycar`) -> PHILIPP/LIU
