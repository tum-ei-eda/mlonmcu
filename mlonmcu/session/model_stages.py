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
"""Optional model transformations for explicitly configured run pipelines."""

from pathlib import Path
import time

from mlonmcu.artifact import Artifact, ArtifactFormat
from .stage import Stage, StageResult, register_stage


def model_format(artifact, override=None):
    """Determine the format per artifact, rather than per session."""
    value = override or Path(artifact.name).suffix.lstrip(".").lower()
    if value not in ("onnx", "tflite"):
        raise ValueError(f"Unsupported model format: {value}")
    return value


def required_formats(context, key=None, visited=None):
    """Follow model consumers, including quantizers, to their format constraints."""
    run = context.run
    if run is None:
        return set()
    key = context.key if key is None else key
    visited = set() if visited is None else visited
    if key in visited:
        return set()
    visited.add(key)
    constraints = []
    for consumer in run.pipeline.stages[run.pipeline.stages.index(key) + 1 :]:
        spec = run.pipeline.stage_spec(run, consumer)
        binding = spec["inputs"].get("model", "")
        if binding.split(".")[0].lower() != key.name.lower():
            continue
        cfg = spec.get("config", {})
        if cfg.get("input_format"):
            constraints.append({cfg["input_format"]})
        elif spec["uses"] in ("build", "tune"):
            formats = getattr(run.backend, "supported_formats", [])
            if formats:
                constraints.append({getattr(fmt, "name", str(fmt)).lower() for fmt in formats})
        elif spec["uses"] == "quantize":
            constraints.append(required_formats(context, consumer, visited) or {"onnx", "tflite"})
        elif spec["uses"] == "convert":
            constraints.append({"onnx", "tflite"})
    result = set.intersection(*constraints) if constraints else set()
    if constraints and not result:
        raise ValueError("Model consumers require incompatible formats; specify separate conversion branches")
    return result


def calibration_samples(config):
    """Normalize explicit NPZ data or MLonMCU's own validation input artifact."""
    import numpy as np

    if "_calibration_samples" in config:
        samples = config["_calibration_samples"]
    elif config.get("calibration_data_reader"):
        import importlib.util

        spec_text = config["calibration_data_reader"]
        filename, separator, class_name = spec_text.rpartition(":")
        if not separator or not filename or not class_name:
            raise ValueError("calibration_data_reader must be /path/to/reader.py:ClassName")
        spec = importlib.util.spec_from_file_location("mlonmcu_calibration_reader", filename)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load calibration data reader: {filename}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reader = getattr(module, class_name)(**config.get("data_reader_args", {}))
        if not callable(getattr(reader, "get_next", None)):
            raise ValueError("Calibration reader must implement get_next()")
        samples = []
        while True:
            sample = reader.get_next()
            if sample is None:
                break
            samples.append(sample)
    else:
        path = config.get("representative_data")
        if not path or path == "auto":
            raise ValueError("Calibration requires representative_data or reuse_validation_inputs=true")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if not arrays or len({len(array) for array in arrays.values()}) != 1:
            raise ValueError("Calibration inputs must have equal nonzero sample counts")
        samples = [
            {name: array[index] for name, array in arrays.items()} for index in range(len(next(iter(arrays.values()))))
        ]
    mapping = config.get("input_name_map", {})
    mapped = []
    for sample in samples:
        if not isinstance(sample, dict) or not sample:
            raise ValueError("Calibration samples must be nonempty dictionaries of input tensors")
        renamed = {mapping.get(name, name): np.asarray(value) for name, value in sample.items()}
        if len(renamed) != len(sample):
            raise ValueError("input_name_map maps multiple inputs to the same name")
        mapped.append(renamed)
    if not mapped:
        raise ValueError("Calibration dataset is empty")
    return mapped


def layout_permutation(layouts):
    if (
        not isinstance(layouts, (list, tuple))
        or len(layouts) != 2
        or not all(isinstance(layout, str) for layout in layouts)
    ):
        raise ValueError("Layout changes must be [source_layout, target_layout], e.g. [NHWC, NCHW]")
    source, target = layouts
    supported = {"NWC", "NCW", "NHWC", "NCHW", "NDHWC", "NCDHW"}
    if source not in supported or target not in supported or set(source) != set(target):
        raise ValueError(f"Unsupported layout change: {layouts}")
    return [source.index(axis) for axis in target]


def transpose_samples(samples, changes):
    import numpy as np

    result = []
    for sample in samples:
        sample = dict(sample)
        for name, layouts in changes.items():
            if name not in sample:
                raise ValueError(f"Layout input {name} missing from calibration data")
            sample[name] = np.transpose(sample[name], layout_permutation(layouts)).copy()
        result.append(sample)
    return result


def validation_inputs(context, variant):
    """Find input data along this model variant's ancestry, never another run."""
    import io
    import numpy as np

    run = context.run
    ancestor = (context.source("model"), variant)
    layout_changes = []
    while ancestor[0] is not None:
        artifacts = run.artifacts_per_stage.get(ancestor[0], {}).get(ancestor[1], [])
        matches = [artifact for artifact in artifacts if "inputs" in artifact.flags and "npy" in artifact.flags]
        if matches:
            if len(matches) != 1:
                raise ValueError("Ambiguous validation input artifacts")
            artifact = matches[0]
            # Object arrays are the existing, trusted frontend artifact format.
            stream = io.BytesIO(artifact.raw) if artifact.raw is not None else artifact.path
            samples = list(np.load(stream, allow_pickle=True))
            for changes in reversed(layout_changes):
                samples = transpose_samples(samples, changes)
            return samples
        pipeline = getattr(run, "pipeline", None)
        if pipeline is not None:
            spec = pipeline.specs.get(ancestor[0], {})
            if spec.get("uses") == "convert":
                layout_changes.append(spec.get("config", {}).get("input_layouts", {}))
        ancestor = run.sub_parents.get(ancestor, (None, None))
    raise ValueError(
        "No validation inputs found in the model ancestry; enable gen_data/validate_new or supply NPZ data"
    )


class ModelTransformStage(Stage):
    input_names = {"model"}
    options = set()

    def validate(self, context):
        context.require_inputs("model")
        unknown = set(context.config) - self.options - {"output_name", "input_format", "output_format"}
        if unknown:
            raise ValueError(f"Unsupported {context.spec['uses']} options: {sorted(unknown)}")
        name = context.config.get("output_name")
        if name is not None and (not isinstance(name, str) or Path(name).name != name):
            raise ValueError("output_name must be a filename")
        for models in context.inputs["model"].values():
            if len(models) != 1:
                raise ValueError("Expected exactly one model per variant")
            source = model_format(models[0], context.config.get("input_format"))
            target = self.target_format(context, source)
            if name and Path(name).suffix != f".{target}":
                raise ValueError("output_name extension must match the output format")

    def target_format(self, context, source):
        target = context.config.get("output_format")
        if target is not None:
            if target not in ("onnx", "tflite"):
                raise ValueError(f"Unsupported output_format: {target}")
            return target
        if isinstance(self, QuantizeStage):
            return source
        formats = required_formats(context)
        if not formats or source in formats:
            return source
        if source == "tflite" and "onnx" in formats:
            return "onnx"
        raise ValueError(f"No conversion path from {source} to required formats {sorted(formats)}")

    def execute(self, context):
        result = StageResult()
        for index, (variant, models) in enumerate(context.inputs["model"].items()):
            source = model_format(models[0], context.config.get("input_format"))
            target = self.target_format(context, source)
            directory = context.workdir / "outputs" / str(index)
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / context.config.get(
                "output_name", f"{'quantized' if isinstance(self, QuantizeStage) else 'converted'}.{target}"
            )
            output.unlink(missing_ok=True)
            started = time.monotonic()
            config = dict(context.config)
            if (
                isinstance(self, QuantizeStage)
                and not (config.get("representative_data") or config.get("calibration_data_reader"))
                and config.get("reuse_validation_inputs")
                and config.get("mode") != "dynamic"
            ):
                config["_calibration_samples"] = validation_inputs(context, variant)
            elif isinstance(self, QuantizeStage) and config.get("representative_data") == "auto":
                config["_calibration_samples"] = validation_inputs(context, variant)
            self.transform(Path(models[0].path), output, config)
            if not output.is_file():
                raise RuntimeError(f"{context.key.name} did not produce {output}")
            result.artifacts[variant] = [
                Artifact(output.name, path=output, fmt=ArtifactFormat.PATH, flags=["model", target])
            ]
            result.parents[variant] = (context.source("model"), variant)
            result.metrics[variant] = {
                f"{context.key.name.title()} time [s]": time.monotonic() - started,
                "Output model size [bytes]": output.stat().st_size,
            }
        return result


@register_stage("quantize")
class QuantizeStage(ModelTransformStage):
    """Dispatch ONNX dynamic or TFLite calibrated integer quantization."""

    options = {
        "weight_type",
        "per_channel",
        "reduce_range",
        "op_types_to_quantize",
        "representative_data",
        "input_type",
        "output_type",
        "activation_type",
        "allow_float",
        "mode",
        "reuse_validation_inputs",
        "input_name_map",
        "quant_format",
        "calibration_method",
        "calibration_data_reader",
        "data_reader_args",
    }

    def validate(self, context):
        super().validate(context)
        for key in ("per_channel", "reduce_range", "allow_float", "reuse_validation_inputs"):
            if not isinstance(context.config.get(key, False), bool):
                raise ValueError(f"{key} must be a boolean")
        if context.config.get("mode", "auto") not in ("auto", "dynamic", "static"):
            raise ValueError("mode must be auto, dynamic, or static")
        if context.config.get("representative_data") and context.config.get("calibration_data_reader"):
            raise ValueError("Choose representative_data or calibration_data_reader")
        if not isinstance(context.config.get("data_reader_args", {}), dict):
            raise ValueError("data_reader_args must be a mapping")
        mapping = context.config.get("input_name_map", {})
        if not isinstance(mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
        ):
            raise ValueError("input_name_map must map source input names to target input names")
        for models in context.inputs["model"].values():
            fmt = model_format(models[0], context.config.get("input_format"))
            if self.target_format(context, fmt) != fmt:
                raise ValueError("QUANTIZE preserves format; insert CONVERT to change formats")
            cfg = context.config
            if fmt == "tflite":
                if cfg.get("weight_type", "int8") not in ("int8", "QInt8"):
                    raise ValueError("TFLite calibrated quantization supports int8 weights")
                if set(cfg) & {"reduce_range", "op_types_to_quantize"}:
                    raise ValueError("reduce_range and op_types_to_quantize apply only to ONNX")
            elif set(cfg) & {"input_type", "output_type", "allow_float"}:
                raise ValueError("input_type, output_type, and allow_float options apply only to TFLite")
            if fmt == "tflite" and not (
                cfg.get("representative_data")
                or cfg.get("reuse_validation_inputs")
                or cfg.get("calibration_data_reader")
            ):
                raise ValueError("TFLite quantization requires representative_data (NPZ)")

    def transform(self, source, output, config):
        fmt = config.get("input_format", source.suffix.lstrip(".").lower())
        if fmt == "tflite":
            return self.quantize_tflite(source, output, config)
        try:
            from onnxruntime.quantization import (
                QuantType,
                QuantFormat,
                CalibrationMethod,
                CalibrationDataReader,
                quantize_dynamic,
                quantize_static,
            )
        except ImportError as exc:
            raise RuntimeError("QUANTIZE requires onnxruntime; install mlonmcu[onnx]") from exc
        dtype = config.get("weight_type", "QInt8")
        aliases = {"int8": "QInt8", "uint8": "QUInt8"}
        try:
            weight_type = QuantType[aliases.get(dtype, dtype)]
        except KeyError as exc:
            raise ValueError(f"Unsupported ONNX weight_type {dtype}; available: {list(QuantType.__members__)}") from exc
        calibrated = bool(
            config.get("representative_data")
            or config.get("_calibration_samples")
            or config.get("calibration_data_reader")
        )
        mode = config.get("mode", "auto")
        mode = ("static" if calibrated else "dynamic") if mode == "auto" else mode
        kwargs = dict(
            weight_type=weight_type,
            per_channel=config.get("per_channel", False),
            reduce_range=config.get("reduce_range", False),
            op_types_to_quantize=config.get("op_types_to_quantize"),
        )
        if mode == "dynamic":
            if calibrated or set(config) & {"activation_type", "quant_format", "calibration_method"}:
                raise ValueError("Dynamic ONNX quantization does not consume calibration data or static options")
            quantize_dynamic(str(source), str(output), **kwargs)
        else:
            samples = calibration_samples(config)

            class Reader(CalibrationDataReader):
                def __init__(self):
                    self.rewind()

                def get_next(self):
                    return next(self.iterator, None)

                def rewind(self):
                    self.iterator = iter(samples)

            try:
                activation = config.get("activation_type", "QInt8")
                activation = QuantType[aliases.get(activation, activation)]
                quant_format = QuantFormat[config.get("quant_format", "QDQ")]
                method = CalibrationMethod[config.get("calibration_method", "MinMax")]
            except KeyError as exc:
                raise ValueError(f"Unsupported ONNX static quantization option: {exc}") from exc
            quantize_static(
                str(source),
                str(output),
                Reader(),
                activation_type=activation,
                quant_format=quant_format,
                calibrate_method=method,
                **kwargs,
            )

    def quantize_tflite(self, source, output, config):
        import tensorflow as tf

        # TensorFlow exposes no public API to requantize an existing flatbuffer.
        from tensorflow.lite.python.optimize.calibrator import Calibrator, add_intermediate_tensors

        interpreter = tf.lite.Interpreter(model_path=str(source))
        inputs = interpreter.get_input_details()
        if config.get("mode", "auto") == "dynamic":
            raise ValueError("TFLite adapter supports static calibrated quantization only")
        samples_data = calibration_samples(config)
        names = [item["name"] for item in inputs]
        for sample in samples_data:
            if set(sample) != set(names):
                raise ValueError("Calibration keys must match TFLite input names; use input_name_map after conversion")

        def samples():
            for sample in samples_data:
                yield [sample[name] for name in names]

        def dtype(key, default):
            try:
                return tf.dtypes.as_dtype(config.get(key, default))
            except TypeError as exc:
                raise ValueError(f"Unsupported {key}: {config.get(key)}") from exc

        content = Calibrator(add_intermediate_tensors(source.read_bytes())).calibrate_and_quantize(
            samples,
            dtype("input_type", "int8"),
            dtype("output_type", "int8"),
            config.get("allow_float", False),
            activations_type=dtype("activation_type", "int8"),
            disable_per_channel=not config.get("per_channel", True),
        )
        output.write_bytes(content)


@register_stage("convert")
class ConvertStage(ModelTransformStage):
    """Select conversion per model and consumer; preserve already suitable formats."""

    options = {"opset", "input_layouts", "output_layouts"}

    def validate(self, context):
        super().validate(context)
        for key in ("input_layouts", "output_layouts"):
            changes = context.config.get(key, {})
            if not isinstance(changes, dict) or not all(isinstance(name, str) for name in changes):
                raise ValueError(f"{key} must map tensor names to layout pairs")
            for layouts in changes.values():
                layout_permutation(layouts)
        if context.config.get("input_layouts") or context.config.get("output_layouts"):
            for models in context.inputs["model"].values():
                source = model_format(models[0], context.config.get("input_format"))
                if self.target_format(context, source) != "onnx":
                    raise ValueError("Layout rewriting currently requires ONNX output")
        opset = context.config.get("opset", 15)
        if type(opset) is not int or opset < 1:
            raise ValueError("opset must be a positive integer")

    def transform(self, source, output, config):
        import shutil

        source_format = config.get("input_format", source.suffix.lstrip(".").lower())
        if source_format == output.suffix.lstrip("."):
            shutil.copyfile(source, output)
            self.rewrite_layouts(output, config)
            return
        if (source_format, output.suffix) != ("tflite", ".onnx"):
            raise ValueError(f"No conversion path from {source_format} to {output.suffix}")
        try:
            from tf2onnx.convert import from_tflite
        except ImportError as exc:
            raise RuntimeError("CONVERT requires tf2onnx and TensorFlow; install mlonmcu[tflite2onnx]") from exc
        from_tflite(str(source), opset=config.get("opset", 15), output_path=str(output))
        self.rewrite_layouts(output, config)

    def rewrite_layouts(self, output, config):
        inputs, outputs = config.get("input_layouts", {}), config.get("output_layouts", {})
        if not inputs and not outputs:
            return
        if output.suffix != ".onnx":
            raise ValueError("Layout rewriting currently requires ONNX output")
        import onnx
        from onnx import helper

        model = onnx.load(output)
        graph = model.graph
        if set(inputs) & set(outputs):
            raise ValueError("Input/output layout changes must reference distinct tensor names")
        occupied = {name for node in graph.node for name in list(node.input) + list(node.output)}
        occupied.update(value.name for value in list(graph.input) + list(graph.output) + list(graph.initializer))

        def internal_name(name):
            candidate = name + "__layout_internal"
            while candidate in occupied:
                candidate += "_"
            occupied.add(candidate)
            return candidate

        def rename_uses(nodes, old, new):
            for node in nodes:
                for index, name in enumerate(node.input):
                    if name == old:
                        node.input[index] = new
                for attr in node.attribute:
                    nested = [attr.g] if attr.type == onnx.AttributeProto.GRAPH else attr.graphs
                    for subgraph in nested:
                        local = {value.name for value in list(subgraph.input) + list(subgraph.initializer)}
                        if old not in local:
                            rename_uses(subgraph.node, old, new)

        prefix, suffix = [], []
        for changes, boundaries, is_input in ((inputs, graph.input, True), (outputs, graph.output, False)):
            by_name = {value.name: value for value in boundaries}
            for name, layouts in changes.items():
                if name not in by_name:
                    raise ValueError(f"Unknown {'input' if is_input else 'output'} tensor: {name}")
                value = by_name[name]
                dims = value.type.tensor_type.shape.dim
                perm = layout_permutation(layouts)
                if len(dims) != len(perm):
                    raise ValueError(f"Layout rank does not match tensor {name}")
                if any(initializer.name == name for initializer in graph.initializer):
                    raise ValueError(f"Cannot rewrite initializer boundary: {name}")
                if is_input and any(item.name == name for item in graph.output):
                    raise ValueError("Direct input-to-output aliases require distinct boundary names")
                internal = internal_name(name)
                rename_uses(graph.node, name, internal)
                if not is_input:
                    for node in graph.node:
                        for index, produced in enumerate(node.output):
                            if produced == name:
                                node.output[index] = internal
                for info in graph.value_info:
                    if info.name == name:
                        info.name = internal
                old_dims = list(dims)
                del dims[:]
                dims.extend(old_dims[index] for index in perm)
                if is_input:
                    inverse = [perm.index(index) for index in range(len(perm))]
                    prefix.append(helper.make_node("Transpose", [name], [internal], perm=inverse))
                else:
                    suffix.append(helper.make_node("Transpose", [internal], [name], perm=perm))
        nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(prefix + nodes + suffix)
        onnx.checker.check_model(model)
        onnx.save(model, output)
