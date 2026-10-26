"""Validation input generation and conversion regressions."""

import numpy as np
import pytest

from mlonmcu.models.data_utils import generate_input_data


@pytest.mark.parametrize("fill", ["zeros", "ones", "random"])
@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_synthetic_inputs_use_tensor_dtype(fill, dtype):
    data = generate_input_data(
        ["x"], {"x": dtype}, {"x": [2, 3]}, {}, {"x": [0.5, -128, "float32", [0, 1]]}, [], fill_mode=fill, number=3
    )
    assert len(data) == 3
    assert all(sample["x"].shape == (2, 3) and sample["x"].dtype == dtype for sample in data)


@pytest.mark.parametrize(
    "dtype,quant,raw,expected",
    [
        ("int8", [1.0, -128, "uint8", [0, 255]], bytes([0, 127, 128, 255]), [-128, -1, 0, 127]),
        (
            "int8",
            [0.5, 0, "float32", None],
            np.array([-100, -1, 1, 100], dtype="float32").tobytes(),
            [-128, -2, 2, 127],
        ),
        ("float32", None, np.array([0, 1, 2, 3], dtype="float32").tobytes(), [0, 1, 2, 3]),
    ],
)
def test_file_inputs_quantize_without_overflow(tmp_path, dtype, quant, raw, expected):
    (tmp_path / "0.bin").write_bytes(raw)
    # Files outside the requested sample count must not be read or parsed as tensors.
    (tmp_path / "1.bin").write_bytes(b"invalid")
    data = generate_input_data(["x"], {"x": dtype}, {"x": [4]}, {}, {"x": quant} if quant else {}, [tmp_path], number=1)
    assert len(data) == 1
    np.testing.assert_array_equal(data[0]["x"], np.array(expected, dtype=dtype))


@pytest.mark.parametrize("mode", ["file", "model"])
def test_frontend_validation_artifacts_use_data_helpers(tmp_path, mode):
    """Keep frontend configuration and callback wiring covered after extraction."""
    import io

    from mlonmcu.feature.features import get_available_features
    from mlonmcu.models.frontend import TfLiteFrontend
    from mlonmcu.models.model import Model

    (tmp_path / "model.tflite").write_bytes(b"model")
    inputs = tmp_path / "input"
    outputs = tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    for sample in range(2):
        np.array([[sample + 1]], dtype="float32").tofile(inputs / f"{sample}.bin")
        np.array([[sample + 1, -sample - 1]], dtype="float32").tofile(outputs / f"{sample}.bin")
    (tmp_path / "labels.csv").write_text("i,label_idx\n1,1\n0,0\n")
    model = Model("model", tmp_path / "model.tflite")
    model.metadata = {
        "network_parameters": {
            "input_nodes": [{"name": "x", "shape": [1, 1], "dtype": "float32", "example_input": {"path": "input"}}],
            "output_nodes": [
                {
                    "name": "y",
                    "shape": [1, 2],
                    "dtype": "float32",
                    "test_output_path": "output",
                    "test_labels_file": "labels.csv",
                }
            ],
        }
    }
    config = {"gen_data.number": 2, "gen_ref_data.mode": mode, "gen_ref_labels.mode": mode}
    registry = get_available_features()
    features = [registry[name](config=config) for name in ["gen_data", "gen_ref_data", "gen_ref_labels"]]
    frontend = TfLiteFrontend(features=features)
    calls = []

    def inference(current_model, data, quant=False, dequant=False):
        assert current_model is model
        assert not quant and dequant
        calls.append(data)
        return {"y": np.concatenate([data["x"], -data["x"]], axis=1)}

    frontend.inference = inference
    artifacts = {artifact.name: artifact for artifact in frontend.process_metadata(model, cfg={})}
    assert "model_info.yml" in artifacts
    data = np.load(io.BytesIO(artifacts["inputs.npy"].raw), allow_pickle=True)
    refs = np.load(io.BytesIO(artifacts["outputs_ref.npy"].raw), allow_pickle=True)
    labels = np.load(io.BytesIO(artifacts["labels_ref.npy"].raw), allow_pickle=True)
    assert len(data) == len(refs) == len(labels) == 2
    np.testing.assert_array_equal(refs[1]["y"], [[2, -2]])
    np.testing.assert_array_equal(labels, [0, 1] if mode == "file" else [0, 0])
    assert len(calls) == (0 if mode == "file" else 4)
