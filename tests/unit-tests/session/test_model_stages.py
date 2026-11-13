#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Real model transformations, including delivery of the result to BUILD."""

from types import SimpleNamespace

import numpy as np
import pytest

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.models.model import Model
from mlonmcu.session.run import Run
from mlonmcu.session.stage import StageContext, stage_key
from mlonmcu.session.model_stages import QuantizeStage, ConvertStage


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("via_feature", [False, True])
def test_real_convert_quantize_build(tmp_path, calibrated, via_feature):
    tf = pytest.importorskip("tensorflow")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("tf2onnx")

    class Network(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="input")])
        def predict(self, x):
            return tf.matmul(x, tf.constant(np.arange(12, dtype=np.float32).reshape(4, 3)))

    network = Network()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([network.predict.get_concrete_function()], network)
    source = tmp_path / "network.tflite"
    source.write_bytes(converter.convert())

    onnx_source = tmp_path / "network.onnx"
    from onnx import helper, numpy_helper, TensorProto

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "mixed",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3), "w")],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    proto.ir_version = 8
    onnx.save(proto, onnx_source)

    class Frontend:
        name = "test"
        config = {}
        DEFAULTS = {}

        def generate_artifacts(self, model):
            import io

            result = {}
            for variant, path, name in (
                ("tflite", source, tf.lite.Interpreter(model_path=str(source)).get_input_details()[0]["name"]),
                ("onnx", onnx_source, "x"),
            ):
                stream = io.BytesIO()
                np.save(stream, [{name: np.ones((1, 4), np.float32) * value} for value in (0.1, 0.5, 1.0)])
                result[variant] = [
                    Artifact(path.name, path=path, fmt=ArtifactFormat.PATH, flags=["model"]),
                    Artifact("inputs.npy", raw=stream.getvalue(), fmt=ArtifactFormat.RAW, flags=["inputs", "npy"]),
                ]
            return result

        def process_metadata(self, model, cfg):
            return []

    built = []

    class Backend:
        name = "test"
        config = {}
        DEFAULTS = {}
        needs_target = False
        supported_formats = ["onnx"]

        def reconfigure(self):
            pass

        def load_model(self, model, **kwargs):
            built.append(onnx.load(model))

        def generate_artifacts(self):
            return []

    from mlonmcu.feature.features import get_available_features

    feature = get_available_features()["quant_onnx"](
        config={
            "quant_onnx.mode": "static" if calibrated else "dynamic",
            "quant_onnx.reuse_validation_inputs": calibrated,
        }
    )
    quant_id = "quant_onnx_load" if via_feature else "quantize"
    run = Run(
        features=[feature] if via_feature else [],
        model=Model("network", [source]),
        frontends=[Frontend()],
        backend=Backend(),
        framework=SimpleNamespace(name="test", config={}, DEFAULTS={}),
        pipeline=(
            ["load", "build"]
            if via_feature
            else [
                "load",
                {"id": "convert", "inputs": {"model": "load.model"}},
                {
                    "id": "quantize",
                    "inputs": {"model": "convert.model"},
                    "config": (
                        {"reuse_validation_inputs": True, "mode": "static"} if calibrated else {"mode": "dynamic"}
                    ),
                },
                {"id": "build", "inputs": {"model": "quantize.model"}},
            ]
        ),
    )
    run.dir = tmp_path
    run.process(until="done")
    assert not run.failing, run.reason
    assert len(built) == 2
    onnx.checker.check_model(built[0])
    assert any(node.op_type == ("QuantizeLinear" if calibrated else "MatMulInteger") for node in built[0].graph.node)
    output = run.artifacts_per_stage[stage_key(quant_id)]["tflite"][0].path
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: np.ones((1, 4), np.float32)})[0]
    np.testing.assert_allclose(actual, [[18, 22, 26]], rtol=0.02)
    for variant in ("tflite", "onnx"):
        assert run.sub_parents[(stage_key("build"), variant)] == (stage_key(quant_id), variant)


@pytest.mark.parametrize(
    "handler,suffix,config,error",
    [
        (QuantizeStage, "onnx", {"per_channel": "false"}, "boolean"),
        (ConvertStage, "tflite", {"opset": 0}, "opset"),
        (ConvertStage, "txt", {}, "Unsupported model format"),
        (QuantizeStage, "onnx", {"output_name": "../bad.onnx"}, "filename"),
    ],
)
def test_invalid_configuration(tmp_path, handler, suffix, config, error):
    context = StageContext(
        None,
        {"id": "transform", "uses": "transform", "config": config},
        {"model": {"default": [Artifact(f"model.{suffix}", raw=b"x", fmt=ArtifactFormat.RAW)]}},
        tmp_path,
    )
    with pytest.raises(ValueError, match=error):
        handler().validate(context)


@pytest.mark.parametrize("data_source", [None, "npz", "reader"])
def test_quantize_onnx_variants(tmp_path, data_source):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import helper, numpy_helper, TensorProto

    source = tmp_path / "float.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(np.ones((4, 3), np.float32), "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    onnx.save(model, source)
    artifact = Artifact(source.name, path=source, fmt=ArtifactFormat.PATH, flags=["model"])
    context = StageContext(
        None,
        {"id": "quantize", "uses": "quantize", "inputs": {"model": "load.model"}},
        {"model": {"first": [artifact], "second": [artifact]}},
        tmp_path,
    )
    if data_source:
        config = {"mode": "static"}
        if data_source == "npz":
            data_path = tmp_path / "samples.npz"
            np.savez(data_path, x=np.ones((3, 1, 4), np.float32))
            config["representative_data"] = str(data_path)
        else:
            reader_path = tmp_path / "reader.py"
            reader_path.write_text(
                "from onnxruntime.quantization import CalibrationDataReader\n"
                "import numpy as np\n"
                "class Reader(CalibrationDataReader):\n"
                "    def __init__(self, count): self.count = count\n"
                "    def get_next(self):\n"
                "        if self.count == 0: return None\n"
                "        self.count -= 1\n"
                "        return {'x': np.ones((1, 4), np.float32)}\n"
            )
            config.update(calibration_data_reader=f"{reader_path}:Reader", data_reader_args={"count": 3})
        context.spec["config"] = config
    handler = QuantizeStage()
    handler.validate(context)
    result = handler.execute(context)
    for variant in ("first", "second"):
        output = onnx.load(result.artifacts[variant][0].path)
        onnx.checker.check_model(output)
        assert any(node.op_type == ("QuantizeLinear" if data_source else "MatMulInteger") for node in output.graph.node)
        assert result.parents[variant] == (stage_key("load"), variant)
    assert result.artifacts["first"][0].path != result.artifacts["second"][0].path


def test_runtime_format_selection(tmp_path):
    from mlonmcu.session.stage import Pipeline
    from mlonmcu.models.model import ModelFormats

    pipeline = Pipeline(
        [
            "load",
            {"id": "convert", "inputs": {"model": "load.model"}},
            {"id": "quantize", "inputs": {"model": "convert.model"}, "config": {"input_format": "onnx"}},
            {"id": "build", "inputs": {"model": "quantize.model"}},
        ]
    )
    run = SimpleNamespace(
        pipeline=pipeline, backend=SimpleNamespace(supported_formats=[ModelFormats.ONNX]), has_stage=lambda key: True
    )
    context = StageContext(run, pipeline.specs[stage_key("convert")], {}, tmp_path)
    assert ConvertStage().target_format(context, "tflite") == "onnx"
    assert ConvertStage().target_format(context, "onnx") == "onnx"
    pipeline.specs[stage_key("quantize")]["config"] = {}
    run.backend.supported_formats = [ModelFormats.TFLITE]
    assert ConvertStage().target_format(context, "tflite") == "tflite"
    context.spec["config"] = {"output_format": "onnx"}
    assert ConvertStage().target_format(context, "tflite") == "onnx"


@pytest.mark.parametrize("automatic", [False, True, "feature"])
def test_real_tflite_quantization(tmp_path, automatic):
    tf = pytest.importorskip("tensorflow")

    class Network(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
        def predict(self, x):
            return tf.matmul(x, tf.constant(np.arange(12, dtype=np.float32).reshape(4, 3)))

    net = Network()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([net.predict.get_concrete_function()], net)
    source = tmp_path / "source.tflite"
    source.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(source))
    data = tmp_path / "calibration.npz"
    np.savez(
        data, **{interpreter.get_input_details()[0]["name"]: np.linspace(0, 1, 40, dtype=np.float32).reshape(10, 1, 4)}
    )
    output = tmp_path / "quantized.tflite"
    config = {"input_type": "uint8", "output_type": "int8"}
    if automatic:
        import io

        samples = [
            {interpreter.get_input_details()[0]["name"]: sample}
            for sample in np.linspace(0, 1, 40, dtype=np.float32).reshape(10, 1, 4)
        ]
        stream = io.BytesIO()
        np.save(stream, samples)
        inputs_artifact = Artifact("inputs.npy", raw=stream.getvalue(), fmt=ArtifactFormat.RAW, flags=["inputs", "npy"])
        artifact = Artifact(source.name, path=source, fmt=ArtifactFormat.PATH, flags=["model"])
        run = SimpleNamespace(
            artifacts_per_stage={stage_key("load"): {"default": [artifact, inputs_artifact]}}, sub_parents={}
        )
        context = StageContext(
            run,
            {
                "id": "quantize",
                "uses": "quantize",
                "inputs": {"model": "load.model"},
                "config": {**config, "reuse_validation_inputs": True},
            },
            {"model": {"default": [artifact]}},
            tmp_path,
        )
        handler = QuantizeStage()
        handler.validate(context)
        output = handler.execute(context).artifacts["default"][0].path
        if automatic == "feature":
            from mlonmcu.feature.features import get_available_features

            feature_config = {f"quant_tflite.{key}": value for key, value in config.items()}
            feature_config["quant_tflite.reuse_validation_inputs"] = True
            feature = get_available_features()["quant_tflite"](config=feature_config)
            run = Run(features=[feature], pipeline=["load"])
            run.dir = tmp_path
            run.artifacts_per_stage[stage_key("load")] = {"default": [artifact, inputs_artifact]}
            run.completed[stage_key("load")] = True
            run.sub_parents[(stage_key("load"), "default")] = (None, None)
            run.execute_stage(stage_key("quant_tflite_convert_load"))
            output = run.execute_stage(stage_key("quant_tflite_load")).artifacts["default"][0].path
    else:
        QuantizeStage().transform(source, output, {**config, "representative_data": str(data)})
    quantized = tf.lite.Interpreter(model_path=str(output))
    quantized.allocate_tensors()
    inp, out = quantized.get_input_details()[0], quantized.get_output_details()[0]
    assert inp["dtype"] == np.uint8
    assert out["dtype"] == np.int8
    value = np.full((1, 4), 0.5, np.float32)
    scale, zero = inp["quantization"]
    quantized.set_tensor(inp["index"], np.round(value / scale + zero).astype(np.uint8))
    quantized.invoke()
    scale, zero = out["quantization"]
    actual = (quantized.get_tensor(out["index"]).astype(np.float32) - zero) * scale
    np.testing.assert_allclose(actual, [[9, 11, 13]], atol=0.3)


def test_calibration_mapping_and_errors(tmp_path):
    from mlonmcu.session.model_stages import calibration_samples, validation_inputs

    samples = [{"old": np.ones((1, 4), np.float32)}]
    result = calibration_samples({"_calibration_samples": samples, "input_name_map": {"old": "new"}})
    assert list(result[0]) == ["new"]
    with pytest.raises(ValueError, match="empty"):
        calibration_samples({"_calibration_samples": []})
    with pytest.raises(ValueError, match="same name"):
        calibration_samples(
            {"_calibration_samples": [{"a": np.ones(1), "b": np.ones(1)}], "input_name_map": {"a": "x", "b": "x"}}
        )
    context = StageContext(
        SimpleNamespace(artifacts_per_stage={}, sub_parents={}),
        {"id": "quantize", "inputs": {"model": "load.model"}},
        {},
        tmp_path,
    )
    with pytest.raises(ValueError, match="No validation inputs"):
        validation_inputs(context, "default")


def test_repository_dummy_reader():
    from pathlib import Path
    from mlonmcu.session.model_stages import calibration_samples

    pytest.importorskip("onnxruntime")
    path = Path(__file__).resolve().parents[3] / "scripts" / "my_data_reader.py"
    samples = calibration_samples({"calibration_data_reader": f"{path}:DummyDataReader"})
    assert len(samples) == 1
    assert samples[0]["serving_default_input:0"].shape == (1, 1960)
    assert samples[0]["serving_default_input:0"].dtype == np.float32


@pytest.mark.parametrize("layouts,shape", [(["NHWC", "NCHW"], [1, 2, 3, 4]), (["NCHW", "NHWC"], [1, 4, 2, 3])])
def test_onnx_layout_roundtrip(tmp_path, layouts, shape):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import helper, TensorProto, numpy_helper
    from mlonmcu.session.model_stages import layout_permutation

    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "weights"], ["y"])],
        "layout",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    source, output = tmp_path / "source.onnx", tmp_path / "converted.onnx"
    onnx.save(model, source)
    config = {"input_layouts": {"x": layouts}, "output_layouts": {"y": layouts}}
    ConvertStage().transform(source, output, config)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    perm = layout_permutation(layouts)
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = original.transpose(perm)
    assert session.get_inputs()[0].shape == list(expected.shape)
    actual = session.run(None, {"x": expected})[0]
    np.testing.assert_array_equal(actual, (original * 2 + 1).transpose(perm))
    restored = tmp_path / "restored.onnx"
    ConvertStage().transform(
        output, restored, {"input_layouts": {"x": layouts[::-1]}, "output_layouts": {"y": layouts[::-1]}}
    )
    restored_session = ort.InferenceSession(str(restored), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(restored_session.run(None, {"x": original})[0], original * 2 + 1)


def test_tflite_to_onnx_layout(tmp_path):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    ort = pytest.importorskip("onnxruntime")

    class Network(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 2, 3, 4], tf.float32, name="x")])
        def predict(self, x):
            return x * 2

    net = Network()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([net.predict.get_concrete_function()], net)
    source = tmp_path / "source.tflite"
    source.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(source))
    input_name, output_name = interpreter.get_input_details()[0]["name"], interpreter.get_output_details()[0]["name"]
    output = tmp_path / "converted.onnx"
    ConvertStage().transform(
        source,
        output,
        {"input_layouts": {input_name: ["NHWC", "NCHW"]}, "output_layouts": {output_name: ["NHWC", "NCHW"]}},
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    original = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    actual = session.run(None, {input_name: original.transpose(0, 3, 1, 2)})[0]
    np.testing.assert_array_equal(actual, (original * 2).transpose(0, 3, 1, 2))


def test_layout_calibration_ancestry(tmp_path):
    import io
    from mlonmcu.session.stage import Pipeline
    from mlonmcu.session.model_stages import validation_inputs

    original = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    stream = io.BytesIO()
    np.save(stream, [{"x": original}])
    artifact = Artifact("inputs.npy", raw=stream.getvalue(), fmt=ArtifactFormat.RAW, flags=["inputs", "npy"])
    pipeline = Pipeline(
        [
            "load",
            {"id": "convert", "config": {"input_layouts": {"x": ["NHWC", "NCHW"]}}},
            {"id": "quantize", "inputs": {"model": "convert.model"}},
        ]
    )
    run = SimpleNamespace(
        pipeline=pipeline,
        artifacts_per_stage={stage_key("load"): {"default": [artifact]}},
        sub_parents={(stage_key("convert"), "default"): (stage_key("load"), "default")},
    )
    context = StageContext(run, pipeline.specs[stage_key("quantize")], {}, tmp_path)
    samples = validation_inputs(context, "default")
    np.testing.assert_array_equal(samples[0]["x"], original.transpose(0, 3, 1, 2))


def test_quant_onnx_feature_pipeline_contract():
    from mlonmcu.feature.features import get_available_features
    from mlonmcu.session.stage import Pipeline

    cls = get_available_features()["quant_onnx"]
    run = Run(features=[cls()])
    stages = run.pipeline.serialize()
    run.add_features([cls()], append=False)
    assert run.pipeline.serialize() == stages
    assert run.pipeline.specs[stage_key("tune")]["inputs"]["model"] == "quant_onnx_load.model"
    assert run.pipeline.specs[stage_key("build")]["inputs"]["model"] == "quant_onnx_load.model"
    disabled = Run(features=[cls(config={"quant_onnx.enabled": False})])
    assert disabled.pipeline.serialize() == Pipeline().serialize()
    explicit = Run(
        features=[cls()],
        pipeline=[
            "load",
            {"id": "manual", "uses": "convert", "inputs": {"model": "load.model"}},
            {"id": "build", "inputs": {"model": "manual.model"}},
        ],
    )
    assert explicit.pipeline.specs[stage_key("build")]["inputs"]["model"] == "manual.model"
    run.completed[stage_key("load")] = True
    with pytest.raises(ValueError, match="after stages have completed"):
        other = Run(pipeline=["load", "build"])
        other.completed[stage_key("load")] = True
        other.add_features([cls()])


def test_quant_onnx_feature_initializer_roundtrip(tmp_path):
    from mlonmcu.session.run import RunInitializer
    from mlonmcu.feature.features import get_available_features

    cls = get_available_features()["quant_onnx"]
    run = Run(features=[cls()], pipeline=["load", "build"])
    path = tmp_path / "initializer.yml"
    import yaml

    path.write_text(yaml.safe_dump({"runs": [{"pipeline": run.pipeline.serialize(), "feature_names": ["quant_onnx"]}]}))
    restored = RunInitializer.from_file(path)[0]
    recreated = Run(features=[cls()], pipeline=restored.pipeline.serialize())
    assert recreated.pipeline.serialize() == run.pipeline.serialize()


def test_quant_onnx_cli_config_values():
    from mlonmcu.cli.helper.parse import parse_vars
    from mlonmcu.feature.features import get_available_features

    config = parse_vars(
        [
            "quant_onnx.mode=static",
            "quant_onnx.reuse_validation_inputs=true",
            'quant_onnx.input_layouts={"x": ["NHWC", "NCHW"]}',
            "quant_onnx.per_channel=true",
        ]
    )
    feature = get_available_features()["quant_onnx"](config=config)
    run = Run(features=[feature], pipeline=["load", "build"])
    quant = run.pipeline.specs[stage_key("quant_onnx_load")]["config"]
    convert = run.pipeline.specs[stage_key("quant_onnx_convert_load")]["config"]
    assert quant["mode"] == "static" and quant["per_channel"] and quant["reuse_validation_inputs"]
    assert convert["input_layouts"] == {"x": ["NHWC", "NCHW"]}


def test_quant_tflite_feature_contract(tmp_path):
    from mlonmcu.feature.features import get_available_features

    cls = get_available_features()["quant_tflite"]
    run = Run(features=[cls(config={"quant_tflite.reuse_validation_inputs": True})])
    pipeline = run.pipeline.serialize()
    run.add_features([cls(config={"quant_tflite.reuse_validation_inputs": True})])
    assert run.pipeline.serialize() == pipeline
    quant = run.pipeline.specs[stage_key("quant_tflite_load")]["config"]
    assert quant["mode"] == "static" and quant["per_channel"]
    assert "reduce_range" not in quant
    assert run.pipeline.specs[stage_key("build")]["inputs"]["model"] == "quant_tflite_load.model"
    with pytest.raises(ValueError, match="mutually exclusive"):
        Run(features=[cls(), get_available_features()["quant_onnx"]()])
    disabled = cls(config={"quant_tflite.enabled": False})
    Run(features=[disabled, get_available_features()["quant_onnx"]()])
    source = tmp_path / "source.onnx"
    source.write_bytes(b"model")
    with pytest.raises(ValueError, match="No conversion path"):
        ConvertStage().transform(source, tmp_path / "output.tflite", {"output_format": "tflite"})


def test_same_format_conversion_preserves_bytes(tmp_path):
    for fmt in ("onnx", "tflite"):
        source = tmp_path / f"source.{fmt}"
        source.write_bytes(b"unchanged model")
        output = tmp_path / f"converted.{fmt}"
        ConvertStage().transform(source, output, {})
        assert output.read_bytes() == source.read_bytes()
