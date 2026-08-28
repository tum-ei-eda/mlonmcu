import sys
from types import SimpleNamespace

import numpy as np

from mlonmcu.models.model_info import (
    MLIRModelInfo,
    ModelInfo,
    ONNXModelInfo,
    RelayModelInfo,
    TensorInfo,
    normalize_shape,
    normalize_shape_dim,
    shape_from_str,
)
from mlonmcu.models.model import parse_shape_string


def test_normalize_shape_supports_static_dynamic_and_symbolic_dimensions():
    assert normalize_shape([1, np.int64(2), -1, "?", "batch", "2 * channels"]) == (
        1,
        2,
        None,
        None,
        "batch",
        "2 * channels",
    )
    assert normalize_shape_dim("42") == 42
    assert normalize_shape_dim("Any") is None


def test_tensor_info_reports_and_resolves_symbolic_shape():
    tensor = TensorInfo("input", ["batch", 3, None, "batch"], "float32")
    assert tensor.shape == ("batch", 3, None, "batch")
    assert tensor.is_dynamic is True
    assert tensor.is_symbolic is True
    assert tensor.symbolic_dims == ("batch",)
    assert tensor.size is None
    assert tensor.resolve_shape({"batch": 2}) == (2, 3, None, 2)
    assert tensor.resolve_shape({"batch": 2}, dynamic_value=4) == (2, 3, 4, 2)
    assert tensor.get_size({"batch": 2}, dynamic_value=4) == 2 * 3 * 4 * 2 * 4


def test_anonymous_dynamic_shape_can_use_a_fallback():
    tensor = TensorInfo("input", [None, 8], "int8")
    assert tensor.is_dynamic is True
    assert tensor.is_symbolic is False
    assert tensor.resolve_shape(dynamic_value=1) == (1, 8)
    assert tensor.get_size(dynamic_value=2) == 16


def test_static_tensor_behavior_is_unchanged():
    tensor = TensorInfo("input/name;0", [2, 3], np.float32, fix_names=True)
    assert tensor.name == "input_name_0"
    assert tensor.is_dynamic is False
    assert tensor.symbolic_dims == ()
    assert tensor.size == 24


def test_model_info_aggregates_symbols_from_inputs_and_outputs():
    info = ModelInfo(
        [TensorInfo("input", ["batch", 3], "float32")],
        [TensorInfo("output", ["batch", "classes"], "float32")],
    )
    assert info.is_dynamic is True
    assert info.symbolic_dims == ("batch", "classes")


def test_shape_from_str_accepts_symbolic_and_dynamic_dimensions():
    assert shape_from_str("batch, 3, ?, -1") == ("batch", 3, None, None)


def test_model_shape_config_accepts_symbolic_and_dynamic_dimensions():
    assert parse_shape_string("input:[batch, 3, ?, -1] state:[1, hidden]") == {
        "input": ["batch", 3, None, None],
        "state": [1, "hidden"],
    }


def test_mlir_model_info_preserves_dynamic_dimensions():
    text = 'func.func @main(%arg0: tensor<?x3xf32> {ml_program.identifier = "image"}) -> tensor<?x2xf32> {'
    info = MLIRModelInfo(text)
    assert info.in_tensors[0].shape == (None, 3)
    assert info.out_tensors[0].shape == (None, 2)
    assert info.is_dynamic is True


def test_relay_model_info_preserves_dynamic_and_symbolic_dimensions():
    text = "def @main(%input: Tensor[(batch, 3, ?), float32]) -> Tensor[(batch, 2), float32] {"
    info = RelayModelInfo(text)
    assert info.in_tensors[0].shape == ("batch", 3, None)
    assert info.out_tensors[0].shape == ("batch", 2)
    assert info.symbolic_dims == ("batch",)


def test_onnx_model_info_preserves_dim_param_and_unknown_dimensions(monkeypatch):
    input_tensor = SimpleNamespace(kind="input")
    output_tensor = SimpleNamespace(kind="output")
    model = SimpleNamespace(graph=SimpleNamespace(input=[input_tensor], output=[output_tensor]))
    tensors = {
        "input": {
            "name": "features",
            "type": {
                "tensorType": {
                    "elemType": 1,
                    "shape": {"dim": [{"dimParam": "batch"}, {"dimValue": "8"}, {}]},
                }
            },
        },
        "output": {
            "name": "probabilities",
            "type": {
                "tensorType": {
                    "elemType": 1,
                    "shape": {"dim": [{"dimParam": "batch"}, {"dimValue": "2"}]},
                }
            },
        },
    }
    monkeypatch.setitem(
        sys.modules,
        "google.protobuf.json_format",
        SimpleNamespace(MessageToDict=lambda tensor: tensors[tensor.kind]),
    )
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=lambda path: model))
    monkeypatch.setitem(
        sys.modules,
        "onnx.helper",
        SimpleNamespace(tensor_dtype_to_np_dtype=lambda elem_type: np.dtype("float32")),
    )

    info = ONNXModelInfo("model.onnx")
    assert info.in_tensors[0].shape == ("batch", 8, None)
    assert info.out_tensors[0].shape == ("batch", 2)
    assert info.out_tensors[0].name == "output"
    assert info.symbolic_dims == ("batch",)
