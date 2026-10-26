"""Validation input generation and conversion regressions."""

import numpy as np
import pytest

from mlonmcu.feature.features import get_available_features

GenData = get_available_features()["gen_data"]
from mlonmcu.models.frontend import TfLiteFrontend


@pytest.mark.parametrize("fill", ["zeros", "ones", "random"])
@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_synthetic_inputs_use_tensor_dtype(fill, dtype):
    feature = GenData(config={"gen_data.fill_mode": fill, "gen_data.number": 3})
    frontend = TfLiteFrontend(features=[feature])
    data = frontend.generate_input_data(
        ["x"], {"x": dtype}, {"x": [2, 3]}, {}, {"x": [0.5, -128, "float32", [0, 1]]}, []
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
    frontend = TfLiteFrontend(features=[GenData(config={"gen_data.number": 1})])
    data = frontend.generate_input_data(["x"], {"x": dtype}, {"x": [4]}, {}, {"x": quant} if quant else {}, [tmp_path])
    assert len(data) == 1
    np.testing.assert_array_equal(data[0]["x"], np.array(expected, dtype=dtype))
