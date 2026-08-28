from unittest.mock import MagicMock

import pytest

from mlonmcu.flow.emx.backend.backend import EMXBackend
from mlonmcu.models.model_info import ModelInfo, TensorInfo


def make_backend():
    return EMXBackend(config={"emx.src_dir": "/tmp/emx"})


def test_emx_compile_args_pin_named_and_anonymous_dimensions():
    backend = make_backend()
    backend.dynamic_input_shapes = {
        "image": ("batch", 3, None),
        "state": ("batch", 8),
    }
    backend.input_shapes = {
        "image": (2, 3, 16),
        "state": (2, 8),
    }
    args = backend.get_emx_compile_args("model.c", "model.onnx")
    assignments = [args[index + 1] for index, arg in enumerate(args) if arg == "--input-dim"]
    assert assignments == ["batch=2", "image:2=16"]


def test_emx_compile_args_reject_conflicting_symbol_values():
    backend = make_backend()
    backend.dynamic_input_shapes = {"left": ("batch", 3), "right": ("batch", 3)}
    backend.input_shapes = {"left": (1, 3), "right": (2, 3)}
    with pytest.raises(AssertionError, match="Conflicting values"):
        backend.get_emx_compile_args("model.c", "model.onnx")


def test_emx_compile_args_reject_rank_mismatch_and_unresolved_values():
    backend = make_backend()
    backend.dynamic_input_shapes = {"input": ("batch", 3)}
    backend.input_shapes = {"input": (1,)}
    with pytest.raises(AssertionError, match="Rank mismatch"):
        backend.get_emx_compile_args("model.c", "model.onnx")

    backend.input_shapes = {"input": ("batch", 3)}
    with pytest.raises(AssertionError, match="must resolve to int"):
        backend.get_emx_compile_args("model.c", "model.onnx")


def test_emx_load_model_applies_concrete_shapes(monkeypatch):
    info = ModelInfo(
        [TensorInfo("input", ("batch", 3), "float32")],
        [TensorInfo("output", ("batch", 2), "float32")],
    )
    monkeypatch.setattr("mlonmcu.flow.emx.backend.backend.get_model_info", MagicMock(return_value=("onnx", info)))
    backend = make_backend()
    backend.load_model(
        "model.onnx",
        input_shapes={"input": (4, 3)},
        output_shapes={"output": (4, 2)},
    )
    assert backend.dynamic_input_shapes == {"input": ("batch", 3)}
    assert backend.model_info.in_tensors[0].shape == (4, 3)
    assert backend.model_info.out_tensors[0].shape == (4, 2)
    assert backend.model_info.is_dynamic is False


def test_emx_load_model_automatically_makes_symbolic_batch_static(monkeypatch):
    info = ModelInfo(
        [TensorInfo("input", ("batch", 3, None), "float32")],
        [TensorInfo("output", ("batch", 2), "float32")],
    )
    monkeypatch.setattr("mlonmcu.flow.emx.backend.backend.get_model_info", MagicMock(return_value=("onnx", info)))
    backend = make_backend()
    backend.load_model("model.onnx")

    assert backend.model_info.in_tensors[0].shape == (1, 3, 1)
    assert backend.model_info.out_tensors[0].shape == (1, 2)
    args = backend.get_emx_compile_args("model.c", "model.onnx")
    assignments = [args[index + 1] for index, arg in enumerate(args) if arg == "--input-dim"]
    assert assignments == ["batch=1", "input:2=1"]


def test_emx_load_model_uses_converted_tflite_shapes_by_position(monkeypatch):
    info = ModelInfo(
        [TensorInfo("converted_input", ("batch", 3, 8, 8), "float32")],
        [TensorInfo("output", ("batch", 2), "float32")],
    )
    monkeypatch.setattr("mlonmcu.flow.emx.backend.backend.get_model_info", MagicMock(return_value=("onnx", info)))
    backend = make_backend()
    backend.load_model("model.onnx", input_shapes={"serving_default_input": (4, 3, 8, 8)})

    assert backend.input_shapes == {"converted_input": (4, 3, 8, 8)}
    assert backend.model_info.out_tensors[0].shape == (4, 2)


def test_emx_dynamic_shape_default_is_configurable(monkeypatch):
    info = ModelInfo(
        [TensorInfo("input", ("batch", 3), "float32")],
        [TensorInfo("output", ("batch", 2), "float32")],
    )
    monkeypatch.setattr("mlonmcu.flow.emx.backend.backend.get_model_info", MagicMock(return_value=("onnx", info)))
    backend = EMXBackend(config={"emx.src_dir": "/tmp/emx", "emx.dynamic_shape_default": 2})
    backend.load_model("model.onnx")

    assert backend.input_shapes == {"input": (2, 3)}
    assert backend.output_shapes == {"output": (2, 2)}


def test_emx_generate_rejects_unresolved_interface_buffers():
    backend = make_backend()
    backend.model = "model.onnx"
    backend.model_info = ModelInfo(
        [TensorInfo("input", ("batch", 3), "float32")],
        [TensorInfo("output", ("batch", 2), "float32")],
    )
    with pytest.raises(AssertionError, match="Provide input_shapes/output_shapes"):
        backend.generate()
