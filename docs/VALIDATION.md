# Output validation (v2)

`validate_new` feeds multiple input samples into a model and saves its output tensors
for comparison in Python. It enables `gen_data`, `gen_ref_data`, `set_inputs`, and
`get_outputs`. Add `--post validate_outputs` to compute comparison metrics. The old
`validate` feature uses model-specific C/C++ checks and remains available for legacy
workflows; do not combine it with `validate_new`.

## Quick start

From the repository root, using the configured workspace:

```sh
source venv/bin/activate
export MLONMCU_HOME=/work/git/mlonmcu/mlonmcu/workspace_default
python -m mlonmcu.cli.main flow run aww --target host_x86 --backend tflmi \
  -f validate_new --post validate_outputs \
  -c gen_data.number=10 -c mlif.batch_size=3 -c mlif.num_threads=4 \
  -c run.export_optional=1
```

Replace `aww` with `vww`, `resnet`, or `toycar`. Use `--backend tvmaotplus`
for the TVM AOT backend when TVM is installed in the workspace.
`gen_data.number` is the maximum number of samples to load. A batch size of three
with ten samples runs the program four times, with one sample in the final batch.
Runtime measurements describe these executions, including transport overhead.

MLIF automatically selects template `v2` when tensor I/O is enabled. Explicitly
requesting `mlif.template_version=v1` with these features raises an error. Other
MLIF runs retain their existing template selection.

## Inputs, references, and quantization

The default modes read the paths declared in each model's `definition.yml`:

- `network_parameters.input_nodes[].example_input.path`: input directory.
- `network_parameters.output_nodes[].test_output_path`: reference output directory.
- `shape` and `dtype`: the model tensor shape and dtype.
- `quantize`: input file dtype/range plus the scale and zero point used to convert
  file data to the model tensor dtype.
- `dequantize`: output scale/zero point and reference file dtype/range.

Raw files are numbered `0.bin`, `1.bin`, etc.; `sample_tensor.bin` is supported for
multiple tensors. Samples must start at zero and be contiguous. Binary files have
no header, and must contain exactly one tensor in the declared storage dtype.
Input data requiring quantization is rounded, saturated to the destination integer range, and cast;
it must not wrap around on overflow. Already quantized data with the same storage
and tensor dtype is copied directly.

In this workspace, all four models already have input and reference data:

| Model | Input files | Reference files | Input storage | Output storage |
| --- | ---: | ---: | --- | --- |
| aww | 10102 | 10102 | float32 | float32 |
| vww | 10993 | 10993 | float32 | float32 |
| resnet | 10000 | 10000 | uint8 | float32 |
| toycar | 10000 | 10000 (generated) | float32 | float32 |

ResNet files contain unsigned image pixels (0–255). Their model inputs are int8;
conversion uses scale 1 and zero point -128. Declaring the file dtype as int8
misinterprets the images. The accompanying metadata patch corrects this.

ToyCar's original `output_tflite` directory duplicates the input spectrograms.
Those files are reconstruction targets, not golden inference outputs, and are
preserved. `output_tflite_generated` contains 10,000 golden outputs generated using
TensorFlow Lite from the supplied model and correctly quantized inputs. Its
`definition.yml` now selects this directory. To reproduce references in a new directory:

```sh
python scripts/generate_validation_references.py workspace_default/models/toycar \
  --number 10000 --output-dir /tmp/toycar-golden
```

The generator refuses to overwrite an existing destination and does not edit metadata.
The original archives are available locally under
`/nas/lrz/home/ga87puy/MLonMCU Data`: `<model>_input.tar (2).xz` and
`<model>_output_tflite.tar.xz`. Inspect archive members before extracting into a
new workspace, and preserve existing model files. Real dataset samples are useful
for accuracy comparisons; synthetic samples only test numerical consistency.

To generate synthetic tensor inputs and compute references using TensorFlow Lite:

```sh
python -m mlonmcu.cli.main flow run resnet --target host_x86 --backend tflmi \
  -f validate_new --post validate_outputs \
  -c gen_data.fill_mode=random -c gen_data.number=10 -c gen_ref_data.mode=model
```

`zeros`, `ones`, and `random` generate values directly in the model tensor dtype.
Model reference generation requires the TensorFlow Python package. The default
file mode does not. Explicit NumPy files and dataset mode are not implemented;
the supported artifact format for this MLIF workflow is `npy`.

## Transport and model support

On MLIF, `auto` chooses `stdin_raw` when stdin is supported, otherwise `rom`.
Output `auto` chooses `stdout_raw` when stdout is supported. Explicit selection:

```sh
# Compile samples into the executable, useful for targets without stdin:
python -m mlonmcu.cli.main flow run aww --target host_x86 --backend tflmi \
  -f validate_new --post validate_outputs -c gen_data.number=10 \
  -c set_inputs.interface=rom -c get_outputs.interface=stdout_raw
```

ROM mode processes all compiled samples in one execution; `mlif.batch_size` does
not split ROM samples across executions. Raw stdin uses the configured batch size.
Binary capture automatically disables live printing. Tensor payload lengths come
from model metadata, so marker bytes inside tensor contents are safe. Truncated
tensors, incomplete samples, and missing/extra output samples raise errors.

Generated v2 support takes precedence over legacy model support directories.
V1 and v2 retain their different callback APIs but share the MLIF CMake setup,
including framework dependencies, compilation settings, and target support.
The MLIF patch is preserved at
`mlonmcu/resources/patches/mlif_validation_v2.patch`; apply it from the mlonmcu-sw
repository root with `git apply`. It is already applied in this workspace.
The ResNet metadata patch is `mlonmcu/resources/patches/resnet_validation_metadata.patch`,
applied from the mlonmcu-models root, and is also already applied here. ToyCar's path change is recorded in
`mlonmcu/resources/patches/toycar_validation_metadata.patch`; generate its new
references before applying that metadata patch to a fresh workspace.

MLIF currently implements ROM/raw-stdin input and raw-stdout output. Filesystem,
text, UART, and RAM transports remain unimplemented. TVM's separate target platform
has its own filesystem transport; these MLIF settings do not configure it.
The raw transport supports multiple output tensors, while model-based reference
inference and some validation metrics still have single-output limitations.

## Metrics and artifacts

The default metrics are `allclose(atol=0.01,rtol=0.01)`, top-1, and top-2 agreement.
Override them with a quoted semicolon-separated list:

```sh
-c 'validate_outputs.validate_metrics=allclose(atol=0.1,rtol=0.1);topk(n=1)'
```

Top-k compares predictions against reference predictions, not ground-truth labels.
For labels use `-f gen_ref_labels --post validate_labels` and the model's label
file. Top-k is not applicable to ToyCar's reconstruction output; use allclose or
`mse(thr=0.1)`. Quantized kernels may differ from desktop TFLite by a few units;
choose tolerances appropriate to the experiment. A successful CLI run means the
workflow completed; inspect metric fractions for agreement, rather than treating
process success as numerical validation. Empty or mismatched sample counts fail.
The postprocess's `report=1` option remains unimplemented.

Exported run artifacts include `model_info.yml`, `inputs.npy`, `outputs_ref.npy`,
`outputs.npy`, and generated `model_support.cpp`. NumPy artifacts contain arrays of
per-sample dictionaries mapping tensor names to arrays. Load trusted artifacts with
`np.load(path, allow_pickle=True)`. Reports appear in the workspace's session and
results directories; `run.export_optional=1` also includes optional diagnostics.

## Verification

See [restoration test results](VALIDATION_RESULTS.md) for the tested backend/model
matrix, numerical agreement fractions, dataset checks, and compatibility runs.

## Historical sources

The implementation from [PR 158](https://github.com/tum-ei-eda/mlonmcu/pull/158)
was merged in September 2024. The earlier
[ValidationNew.md](https://github.com/tum-ei-eda/mlonmcu/blob/refactor-validate/ValidationNew.md)
explains its development, but its commands require manual configuration that is
now automatic and include unfinished examples.

The [tuning_validate_new_ipynb branch](https://github.com/tum-ei-eda/mlonmcu/tree/tuning_validate_new_ipynb)
contains a useful notebook update in commit `2f9144e2`, with host examples using
`validate_new`. The branch also includes unrelated tuning and target changes;
merging the whole branch is unnecessary. Its example has a typo:
`validate_labels.validate_metrics` should be `validate_outputs.validate_metrics`
for output comparisons. This page provides the current command equivalents.
The [mlifio commit](https://github.com/tum-ei-eda/mlonmcu-sw/commit/05aeac7e7284107202f0ef897ca1634ff78442b4)
is already present in the workspace's MLIF history.
