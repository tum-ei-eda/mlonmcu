# Validation restoration checks

Tested on 2026-10-26 with `venv` and `MLONMCU_HOME=workspace_default`, using
`host_x86`, ten samples per model, and raw stdin/stdout in batches of three.
Session numbers below refer to this workspace. Classification agreement compares
against reference predictions. ToyCar uses the regenerated TFLite golden outputs.

| Model | Backend | Samples | Top-1 agreement | Allclose (atol=.01, rtol=.01) | MSE < .1 | Session |
| --- | --- | ---: | --- | --- | --- | ---: |
| aww | tflmi | 10 | 10/10 | 8/10 | 10/10 | 2374 |
| aww | tvmaotplus | 10 | 10/10 | 6/10 | 10/10 | 2378 |
| vww | tflmi | 10 | 10/10 | 10/10 | 10/10 | 2375 |
| vww | tvmaotplus | 10 | 10/10 | 10/10 | 10/10 | 2379 |
| resnet | tflmi | 10 | 10/10 | 7/10 | 10/10 | 2382 |
| resnet | tvmaotplus | 10 | 10/10 | 5/10 | 10/10 | 2380 |
| toycar | tflmi | 10 | N/A | 0/10 | 10/10 | 2385 |
| toycar | tvmaotplus | 10 | N/A | 10/10 | 10/10 | 2386 |

Numerical agreement depends on tolerance and backend quantized kernels. In
particular, TFLM ToyCar differs by quantization units, so the strict `.01` absolute
tolerance fails although every sample has MSE below `.1`. TVM ToyCar matches its
golden outputs at the strict tolerance. These fractions are retained to make the
remaining numerical differences visible.

Additional successful runs:

- Corrected ResNet input storage verified against desktop TFLite: ten exact
  reference matches, versus only four top-1 matches when bytes were read as int8.
- AWW ROM input: five samples, requested batch size two, all five outputs returned
  in one execution; top-1 agreement 5/5 (session 2383).
- Legacy MLIF v1 AWW host compile/run (session 2384).
- Synthetic random ResNet inputs with model-generated references: ten outputs
  returned, top-1 agreement 10/10 on TFLM (session 2387).
- All 10,000 ToyCar golden output files generated in `output_tflite_generated`;
  original reconstruction targets preserved.
- Model input/reference data counts and the first ten samples' sizes and finite
  values checked for all four models.

Unit coverage includes input generation, quantization saturation, unsigned ResNet
pixels, automatic interface/version selection, partial batches, multiple output
tensors, framing markers embedded in payloads, truncated frames, and missing/extra
output samples.

Final unit check: **67 passed, 5 skipped** (`tests/unit-tests/platform`,
`tests/unit-tests/models`, and `tests/unit-tests/session/test_validation_outputs.py`).
