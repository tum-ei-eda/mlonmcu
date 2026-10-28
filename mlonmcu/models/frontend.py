#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
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
import os
import sys
import re
import time
import tempfile
import multiprocessing
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np

from mlonmcu.feature.features import get_matching_features
from mlonmcu.models.model import (
    ModelFormats,
    Model,
    Program,
    ExampleProgram,
    EmbenchProgram,
    EmbenchIoTProgram,
    EmbenchDSPProgram,
    TaclebenchProgram,
    PolybenchProgram,
    CoremarkProgram,
    DhrystoneProgram,
    MathisProgram,
    MibenchProgram,
    OpenASIPProgram,
    RVVBenchProgram,
    ISSBenchProgram,
    CryptoBenchProgram,
    CmsisDSPProgram,
    CmsisNNProgram,
)
from mlonmcu.models.lookup import lookup_models
from mlonmcu.models.data_utils import (
    ensure_model_data,
    generate_input_data,
    generate_output_ref_data,
    generate_ref_labels,
    generate_model_info,
)
from mlonmcu.feature.type import FeatureType
from mlonmcu.config import Configurable, cfg, filter_config, one_of, required, str2bool
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.setup import utils
from mlonmcu.target.metrics import Metrics

from mlonmcu.logging import get_logger

logger = get_logger()


class Frontend(Configurable, ABC):
    FEATURES = {"validate"}

    download_data = cfg(False, cast=str2bool)
    check_integrity = cfg(True, cast=str2bool)
    use_inout_data = cfg(False, cast=str2bool)
    gen_data = cfg(False, cast=str2bool)
    gen_data_file = cfg(None)
    gen_data_fill_mode = cfg(None, cast=one_of(("random", "ones", "zeros", "file", "dataset")), preserve_none=False)
    gen_data_number = cfg(None, cast=int, preserve_none=False)
    gen_data_fmt = cfg(None, cast=one_of(("npy", "npz")), preserve_none=False)
    gen_ref_data = cfg(False, cast=str2bool)
    gen_ref_data_file = cfg(None)
    gen_ref_data_mode = cfg(None, cast=one_of(("file", "model")), preserve_none=False)
    gen_ref_data_fmt = cfg(None, cast=one_of(("npy", "npz")), preserve_none=False)
    gen_ref_labels = cfg(False, cast=str2bool)
    gen_ref_labels_file = cfg(None)
    gen_ref_labels_mode = cfg(None, cast=one_of(("file", "model")), preserve_none=False)
    gen_ref_labels_fmt = cfg(None, cast=one_of(("npy", "npz", "txt", "csv")), preserve_none=False)

    def __init__(self, name, input_formats=None, output_formats=None, features=None, config=None):
        self.name = name
        self.input_formats = input_formats if input_formats else []
        self.output_formats = output_formats if output_formats else []
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)

    def __repr__(self):
        probs = []
        if self.name:
            probs.append(self.name)
        if self.features and len(self.features) > 0:
            probs.append(str(self.features))
        if self.config and len(self.config) > 0:
            probs.append(str(self.config))
        return "Frontend(" + ",".join(probs) + ")"

    def inference(self, model: Model, input_data: Dict[str, np.array]):
        raise NotImplementedError

    def extract_model_info(self, model: Model):
        raise NotImplementedError

    def supports_formats(self, ins=None, outs=None):
        """Returs true if the frontend can handle at least one combination of input and output formats."""
        assert ins is not None or outs is not None, "Please provide a list of input formats, outputs formats or both"
        ret = True
        if ins:
            if not isinstance(ins, list):
                ins = [ins]
            supported = any(fmt in self.input_formats for fmt in ins)
            ret = ret and supported
        if outs:
            if not isinstance(outs, list):
                outs = [outs]
            supported = any(fmt in self.output_formats for fmt in ins)
            ret = ret and supported
        return ret

    def lookup_models(self, names, config=None, context=None):
        return lookup_models(names, frontends=[self], config=config, context=context)

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.FRONTEND)
        for feature in features:
            assert (  # If this assertion occurs, continue with the next frontend instead of failing
                # (TODO: create custom exception type)
                feature.name
                in self.FEATURES
            ), f"Incompatible feature: {feature.name}"
            # Instead we might introduce self.compatible and set it to true at this line
            feature.used = True
            feature.add_frontend_config(self.name, self.config)
            feature.update_formats(self.name, self.input_formats, self.output_formats)
        return features

    @abstractmethod
    # def produce_artifacts(self, model):
    def produce_artifacts(self, model):
        pass

    def process_metadata(self, model, cfg=None):
        if len(model.paths) == 0:
            model_dir = None
            metadata = None
        else:
            model_dir = Path(model.paths[0]).parent.resolve()
            metadata = model.metadata
        if self.check_integrity and metadata is not None:
            network = metadata.get("network") or {}
            checksum = network.get("hash") or {}
            if checksum.get("value") is not None:
                utils.validate_checksum(model.paths[0], checksum["value"], mode=checksum.get("algorithm", "sha1"))
            else:
                logger.debug("Skipping integrity check (missing hash)")
        in_paths = []
        out_paths = []
        labels_paths = []
        input_shapes = {}
        output_shapes = {}
        input_types = {}
        output_types = {}
        input_ranges = {}
        output_ranges = {}
        input_quant_details = {}
        output_quant_details = {}
        needs_inputs = self.use_inout_data or (
            self.gen_data and self.gen_data_fill_mode == "file" and self.gen_data_file == "auto"
        )
        needs_outputs = self.use_inout_data or (
            self.gen_ref_data and self.gen_ref_data_mode == "file" and self.gen_ref_data_file == "auto"
        )
        needs_labels = self.gen_ref_labels and self.gen_ref_labels_mode == "file" and self.gen_ref_labels_file == "auto"
        if metadata is not None and "network_parameters" in metadata:
            network = metadata["network_parameters"]
            assert "input_nodes" in network
            ins = network["input_nodes"]
            for inp in ins:
                name = inp.get("name", None)
                shape = inp.get("shape", None)
                ty = inp.get("dtype", None)
                if ty is None:
                    ty = inp.get("type", None)  # legacy
                rng = inp.get("range", None)
                quantize = inp.get("quantize", None)
                if name and shape:
                    input_shapes[name] = shape
                if name and ty:
                    input_types[name] = ty
                if name and rng:
                    input_ranges[name] = rng
                if name and quantize:
                    quant_scale = quantize.get("scale", None)
                    quant_zero_shift = quantize.get("zero_shift", None)
                    quant_dtype = quantize.get("dtype", None)
                    quant_range = quantize.get("range", None)
                    quant_details = [quant_scale, quant_zero_shift, quant_dtype, quant_range]
                    input_quant_details[name] = quant_details
                if needs_inputs:
                    if "example_input" in inp and "path" in inp["example_input"]:
                        in_data_dir = Path(inp["example_input"]["path"])
                        # TODO: this will only work with relative paths to model dir! (Fallback to parent directories?)
                        in_path = model_dir / in_data_dir
                        in_paths.append(in_path)
            assert "output_nodes" in network
            outs = network["output_nodes"]
            for outp in outs:
                name = outp.get("name", None)
                shape = outp.get("shape", None)
                ty = outp.get("dtype", None)
                if ty is None:
                    ty = outp.get("type", None)  # legacy
                rng = outp.get("range", None)
                dequantize = outp.get("dequantize", None)
                if name and shape:
                    output_shapes[name] = shape
                if name and ty:
                    output_types[name] = ty
                if name and rng:
                    output_ranges[name] = rng
                if name and dequantize:
                    quant_scale = dequantize.get("scale", None)
                    quant_zero_shift = dequantize.get("zero_shift", None)
                    quant_dtype = dequantize.get("dtype", None)
                    quant_range = dequantize.get("range", None)
                    quant_details = [quant_scale, quant_zero_shift, quant_dtype, quant_range]
                    output_quant_details[name] = quant_details
                if needs_outputs:
                    if "test_output_path" in outp:
                        out_data_dir = Path(outp["test_output_path"])
                        out_path = model_dir / out_data_dir
                        out_paths.append(out_path)
                if needs_labels:
                    if "test_labels_file" in outp:
                        labels_file = Path(outp["test_labels_file"])
                        labels_path = model_dir / labels_file
                        labels_paths.append(labels_path)
        elif model_dir is not None:
            fallback_in_path = model_dir / "input"
            if fallback_in_path.is_dir() or (self.download_data and needs_inputs):
                in_paths.append(fallback_in_path)
            fallback_out_path = model_dir / "output"
            if fallback_out_path.is_dir() or (self.download_data and needs_outputs):
                out_paths.append(fallback_out_path)
            fallback_labels_path = model_dir / "output_labels.csv"
            if fallback_labels_path.is_file() or (self.download_data and needs_labels):
                labels_paths.append(fallback_labels_path)
        data_number = self.gen_data_number if self.gen_data else 1
        if self.download_data:
            data_number = ensure_model_data(
                model_dir,
                input_paths=in_paths if needs_inputs and not model.inputs_path else (),
                output_paths=out_paths if needs_outputs and not model.outputs_path else (),
                label_paths=labels_paths if needs_labels and not model.output_labels_path else (),
                number=data_number,
            )
        if model.inputs_path:
            logger.info("Overriding default model input data with user path")
            in_paths = [model.inputs_path]
        if model.outputs_path:
            logger.info("Overriding default model output data with user path")
            out_paths = [model.outputs_path]
        if model.output_labels_path:  # TODO
            logger.info("Overriding default model output labels with user path")
            labels_paths = [model.output_labels_path]

        if needs_inputs:
            for path in in_paths:
                assert path.exists(), f"Input data path does not exist: {path}"
        if needs_outputs:
            for path in out_paths:
                assert path.exists(), f"Output data path does not exist: {path}"
        if needs_labels:
            for path in labels_paths:
                assert path.is_file(), f"Labels file defined in model metadata does not exist: {path}"

        if metadata is not None and "backends" in metadata:
            assert cfg is not None
            backend_options = metadata["backends"]
            for backend in backend_options:
                if backend_options[backend] is not None:
                    flattened = {f"{backend}.{key}": value for key, value in backend_options[backend].items()}
                    cfg.update(flattened)

        if len(input_shapes) > 0:
            assert len(input_types) in [len(input_shapes), 0]
            input_names = list(input_shapes.keys())
        elif len(input_types) > 0:
            input_names = list(input_types.keys())
        else:
            input_names = []

        if metadata is None:
            try:
                (
                    input_names,
                    input_shapes,
                    input_types,
                    input_quant_details,
                    output_names,
                    output_shapes,
                    output_types,
                    output_quant_details,
                ) = self.extract_model_info(model)
            except NotImplementedError:
                logger.warning("Model info could not be extracted.")

        # Detect model support code (Allow overwrite in metadata YAML)
        if model.support_path:
            support_path = model.support_path
        elif model_dir is not None:
            support_path = model_dir / "support"
        else:
            support_path = None
        if support_path is not None and support_path.is_dir():
            assert cfg is not None
            # TODO: onlu overwrite if unset?
            if cfg.get("mlif.model_support_dir", None) is not None:
                cfg.update({"mlif.model_support_dir": support_path})
            # cfg.update({"espidf.model_support_dir": support_path})
            # cfg.update({"zephyr.model_support_dir": support_path})
        if len(in_paths) > 0:
            cfg.update({"mlif.input_data_path": in_paths})
            # cfg.update({"espidf.input_data_path": in_paths})
            # cfg.update({"zephyr.input_data_path": in_paths})
        if len(out_paths) > 0:
            cfg.update({"mlif.output_data_path": out_paths})
            # cfg.update({"espidf.output_data_path": out_paths})
            # cfg.update({"zephyr.output_data_path": out_paths})
        if len(labels_paths) > 0:
            cfg.update({"mlif.output_labels_path": labels_paths})
        if len(input_shapes) > 0:
            cfg.update({f"{model.name}.input_shapes": input_shapes})
        if len(output_shapes) > 0:
            cfg.update({f"{model.name}.output_shapes": output_shapes})
        if len(input_types) > 0:
            cfg.update({f"{model.name}.input_types": input_types})
        if len(output_types) > 0:
            cfg.update({f"{model.name}.output_types": output_types})
        # flattened version
        if len(output_shapes) > 0:
            assert len(output_types) in [len(output_shapes), 0]
            output_names = list(output_shapes.keys())
        elif len(output_shapes) > 0:
            output_names = list(output_types.keys())
        else:
            output_names = []
        artifacts = []
        inputs_data = None
        gen_model_info = True  # TODO: move to self (configurable)
        if gen_model_info:
            model_info_dict = generate_model_info(
                input_names,
                output_names,
                input_shapes,
                output_shapes,
                input_types,
                output_types,
                input_ranges,
                output_ranges,
                input_quant_details,
                output_quant_details,
            )
            import yaml

            content = yaml.dump(model_info_dict)
            model_info_artifact = Artifact(
                "model_info.yml", content=content, fmt=ArtifactFormat.TEXT, flags=("model_info",)
            )
            artifacts.append(model_info_artifact)
        if self.gen_data:
            inputs_data = generate_input_data(
                input_names,
                input_types,
                input_shapes,
                input_ranges,
                input_quant_details,
                in_paths,
                fill_mode=self.gen_data_fill_mode,
                data_file=self.gen_data_file,
                number=data_number,
            )
            fmt = self.gen_data_fmt
            if fmt == "npy":
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tempfilename = Path(tmpdirname) / "inputs.npy"
                    np.save(tempfilename, inputs_data)
                    with open(tempfilename, "rb") as f:
                        raw = f.read()
            elif fmt == "npz":
                raise NotImplementedError
            else:
                raise RuntimeError(f"Unsupported fmt: {fmt}")
            assert raw
            inputs_data_artifact = Artifact(f"inputs.{fmt}", raw=raw, fmt=ArtifactFormat.BIN, flags=("inputs", fmt))
            artifacts.append(inputs_data_artifact)
        if self.gen_ref_data:
            outputs_ref_data = generate_output_ref_data(
                inputs_data,
                model,
                out_paths,
                output_names,
                output_types,
                output_shapes,
                output_quant_details,
                mode=self.gen_ref_data_mode,
                data_file=self.gen_ref_data_file,
                inference=self.inference,
            )
            fmt = self.gen_ref_data_fmt
            if fmt == "npy":
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tempfilename = Path(tmpdirname) / "outputs_ref.npy"
                    np.save(tempfilename, outputs_ref_data)
                    with open(tempfilename, "rb") as f:
                        raw = f.read()
            elif fmt == "npz":
                raise NotImplementedError
            else:
                raise RuntimeError(f"Unsupported fmt: {fmt}")
            assert raw
            outputs_ref_artifact = Artifact(
                f"outputs_ref.{fmt}", raw=raw, fmt=ArtifactFormat.BIN, flags=("outputs_ref", fmt)
            )
            artifacts.append(outputs_ref_artifact)
        if self.gen_ref_labels:
            labels_ref = generate_ref_labels(
                inputs_data,
                model,
                labels_paths,
                mode=self.gen_ref_labels_mode,
                data_file=self.gen_ref_labels_file,
                inference=self.inference,
            )
            fmt = self.gen_ref_labels_fmt
            if fmt == "npy":
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tempfilename = Path(tmpdirname) / "labels.npy"
                    np.save(tempfilename, labels_ref)
                    with open(tempfilename, "rb") as f:
                        raw = f.read()
            elif fmt == "npz":
                raise NotImplementedError
            elif fmt == "txt":
                raise NotImplementedError
            elif fmt == "csv":
                raise NotImplementedError
            else:
                raise RuntimeError(f"Unsupported fmt: {fmt}")
            assert raw
            labels_ref_artifact = Artifact(
                f"labels_ref.{fmt}", raw=raw, fmt=ArtifactFormat.BIN, flags=("labels_ref", fmt)
            )
            artifacts.append(labels_ref_artifact)
        return artifacts

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = []

        if len(model.paths) > 0:
            count = len(model.paths)
        elif len(model.classes) > 0:
            count = len(model.classes)
        else:
            count = 0
        assert count == len(model.formats)
        assert count > 0, f"'{self.name}' frontend expects at least one model"
        max_ins = len(self.input_formats)
        assert count <= max_ins, f"'{self.name}' frontend did not expect more than {max_ins} models"
        formats = model.formats
        assert self.supports_formats(formats), f"Invalid model format for '{self.name}' frontend"

        artifacts = self.produce_artifacts(model)
        if not isinstance(artifacts, list):
            artifacts = [artifacts]
        assert len(artifacts) > 0, f"'{self.name}' frontend should produce at least one model"
        max_outs = len(self.output_formats)
        assert len(artifacts) <= max_outs, f"'{self.name}' frontend should not return more than {max_outs}"

        # If we want to use the same instance of this Frontend in parallel, we need to get rid of self.artifacts...
        return {"default": artifacts}, {}

    def generate_artifacts(self, model) -> List[Artifact]:
        start_time = time.time()
        artifacts, metrics = self.generate(model)
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        if len(metrics) == 0:
            metrics = {"default": Metrics()}
        for name, metrics_ in metrics.items():
            if name == "default":
                metrics_.add("Load Stage Time [s]", diff, True)
            content = metrics_.to_csv(include_optional=True)
            artifact = Artifact("load_metrics.csv", content=content, fmt=ArtifactFormat.TEXT, flags=["metrics"])
            if name not in artifacts:
                artifacts[name] = []
            artifacts[name].append(artifact)
        self.artifacts = artifacts
        return artifacts

    def export_artifacts(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_artifacts() first"

        if not isinstance(path, Path):
            path = Path(path)

        is_dir = len(path.suffix) == 0
        if is_dir:
            assert (
                path.is_dir()
            ), "The supplied path does not exists."  # Make sure it actually exists (we do not create it by default)
            for artifact in self.artifacts:
                artifact.export(path)
        else:
            raise NotImplementedError

    def get_platform_config(self, platform):
        return {}

    def add_platform_config(self, platform, config):
        config.update(self.get_platform_config(platform))

    def get_platform_defs(self, platform):
        return {}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))


class SimpleFrontend(Frontend):
    """An abstract frontend with equivalent input and output formats."""

    # Assumption: only raw model data
    def __init__(self, name, fmt, features=None, config=None):
        super().__init__(
            name,
            input_formats=[fmt],
            output_formats=[fmt],
            features=features,
            config=config,
        )

    def produce_artifacts(self, model):
        assert len(self.input_formats) == len(self.output_formats) == len(model.paths) == 1
        artifacts = []
        name = model.name
        assert "/" not in name
        path = model.paths[0]
        ext = self.input_formats[0].extension
        with open(path, "rb") as handle:  # TODO: is an onnx model raw data or text?
            raw = handle.read()
            artifacts.append(Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"]))
        return artifacts


class MLFFrontend(SimpleFrontend):
    """Frontend for pre-compiled TVM Model Library Format archives."""

    def __init__(self, features=None, config=None):
        super().__init__("mlf", ModelFormats.MLF, features=features, config=config)

    def produce_artifacts(self, model):
        assert len(model.paths) == 1
        path = model.paths[0]
        with open(path, "rb") as handle:
            raw = handle.read()
        return [
            Artifact(
                f"{model.name}.tar",
                raw=raw,
                fmt=ArtifactFormat.MLF,
                flags=["model"],
                archive=True,
            )
        ]


# TODO: move to frontends.py
# TODO: frontend parsed metadata instead of lookup.py?
# TODO: how to find inout_data?
class TfLiteFrontend(SimpleFrontend):
    FEATURES = Frontend.FEATURES | {
        "visualize",
        "split_layers",
        "tflite_analyze",
        "gen_data",
        "gen_ref_data",
        "gen_ref_labels",
    }

    visualize_enable = cfg(False, cast=str2bool)
    visualize_script = cfg(None)
    split_layers = cfg(False, cast=str2bool)
    pack_script = cfg(None)
    analyze_enable = cfg(False, cast=str2bool)
    analyze_script = cfg(None)
    OPTIONAL = {"tflite_analyze.script"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "tflite",
            ModelFormats.TFLITE,
            features=features,
            config=config,
        )

    def extract_model_info(self, model: Model):
        try:
            import tensorflow as tf
        except ImportError as ex:
            logger.error("Missing Python package: tensorflow")
            raise ex

        model_path = str(model.paths[0])
        interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_names = []
        input_shapes = {}
        input_types = {}
        input_quant_details = {}
        output_names = []
        output_shapes = {}
        output_types = {}
        output_quant_details = {}
        for inp in input_details:
            name = str(inp["name"])
            input_names.append(name)
            input_shapes[name] = inp["shape"].tolist()
            input_types[name] = np.dtype(inp["dtype"]).name
            if inp.get("quantization", (0, 0))[0] > 0:
                scale, zero_point = inp["quantization"]
                quant = [scale, zero_point, "float32", None]
                input_quant_details[name] = quant
        for outp in output_details:
            name = str(outp["name"])
            output_names.append(name)
            output_shapes[name] = outp["shape"].tolist()
            output_types[name] = np.dtype(outp["dtype"]).name
            if outp.get("quantization", (0, 0))[0] > 0:
                scale, zero_point = outp["quantization"]
                quant = [scale, zero_point, "float32", None]
                output_quant_details[name] = quant
        return (
            input_names,
            input_shapes,
            input_types,
            input_quant_details,
            output_names,
            output_shapes,
            output_types,
            output_quant_details,
        )

    def inference(self, model: Model, input_data: Dict[str, np.array], quant=False, dequant=False, verbose=False):
        try:
            import tensorflow as tf
        except ImportError as ex:
            logger.error("Missing Python package: tensorflow")
            raise ex

        model_path = str(model.paths[0])
        interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        if verbose:
            print()
            print("Input details:")
            print(input_details)
            print()
            print("Output details:")
            print(output_details)
            print()
        assert len(input_details) == 1, "Multi-inputs not yet supported"
        input_type = input_details[0]["dtype"]
        input_name = input_details[0]["name"]
        input_shape = input_details[0]["shape"]
        assert input_name in input_data, f"Input {input_name} fot found in data"
        np_features = input_data[input_name]
        if quant and input_type == np.int8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            if verbose:
                print("Input scale:", input_scale)
                print("Input zero point:", input_zero_point)
                print()
            np_features = (np_features / input_scale) + input_zero_point
            np_features = np.around(np_features)
        np_features = np_features.astype(input_type)
        np_features = np_features.reshape(input_shape)
        interpreter.set_tensor(input_details[0]["index"], np_features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # If the output type is int8 (quantized model), rescale data
        assert len(output_details) == 1, "Multi-outputs not yet supported"
        output_type = output_details[0]["dtype"]
        output_name = output_details[0]["name"]
        if dequant and output_type == np.int8:
            output_scale, output_zero_point = output_details[0]["quantization"]
            if verbose:
                print("Raw output scores:", output)
                print("Output scale:", output_scale)
                print("Output zero point:", output_zero_point)
                print()
            output = output_scale * (output.astype(np.float32) - output_zero_point)

        if verbose:
            # Print the results of inference
            print("Inference output:", output, type(output))
        return {output_name: output}

    def produce_artifacts(self, model):
        assert len(self.input_formats) == len(model.paths) == 1
        artifacts = []

        name = model.name
        # assert "/" not in name
        if "/" in name:
            name = name.rsplit("/", 1)[-1]
        path = model.paths[0]
        ext = self.input_formats[0].extension
        with open(path, "rb") as handle:
            raw = handle.read()
            artifacts.append(Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"]))

        if self.analyze_enable:
            with tempfile.TemporaryDirectory() as tmpdirname:
                out_file = str(Path(tmpdirname) / "tflite_analyze.csv")

                args = [
                    path,
                    "--csv",
                    out_file,
                    "--ops",
                    "--estimate-macs",
                    "--estimate-rom",
                    "--estimate-ram",
                ]

                assert self.analyze_script is not None
                assert Path(self.analyze_script).is_file(), f"Script {self.analyze_script} not found."
                utils.python(self.analyze_script, *args)

                with open(out_file, "r") as handle:
                    tflite_analyze_csv = handle.read()

                tflite_analyze_artifact = Artifact(
                    "tflite_analyze.csv",
                    content=tflite_analyze_csv,
                    fmt=ArtifactFormat.TEXT,
                )
                artifacts.append(tflite_analyze_artifact)
        if self.visualize_enable:
            assert self.visualize_script is not None

            in_file = model.paths[0]
            ext = "html"
            with tempfile.TemporaryDirectory() as tmpdirname:
                out_file = str(Path(tmpdirname) / f"tflite_visualize.{ext}")

                utils.python(self.visualize_script, in_file, out_file)

                with open(out_file, "r") as handle:
                    tflite_visualize_text = handle.read()

                tflite_visualize_artifact = Artifact(
                    f"tflite_visualize.{ext}",
                    content=tflite_visualize_text,
                    fmt=ArtifactFormat.TEXT,
                )
                artifacts.append(tflite_visualize_artifact)

        if self.visualize_enable and self.analyze_enable:
            assert len(self.output_formats) == 3
        elif self.visualize_enable or self.analyze_enable:
            assert len(self.output_formats) == 2
        else:
            assert len(self.output_formats) == 1

        return artifacts

    def generate(self, model) -> Tuple[dict, dict]:
        if self.split_layers:
            artifacts = {}

            name = model.name
            path = model.paths[0]
            formats = model.formats
            assert self.supports_formats(formats), f"Invalid model format for '{self.name}' frontend"

            ret = self.produce_artifacts(model)
            if not isinstance(ret, list):
                ret = [ret]
            assert len(ret) > 0, f"'{self.name}' frontend should produce at least one model"
            max_outs = len(self.output_formats)
            assert len(ret) <= max_outs, f"'{self.name}' frontend should not return more than {max_outs}"
            artifacts["default"] = ret
            with tempfile.TemporaryDirectory() as tmpdirname:

                def get_num_layers(file):
                    tflite_pack_args = [path, "--count-layers", "--noop"]
                    out = utils.execute(self.pack_script, *tflite_pack_args)
                    matches = re.compile(r"Found\s(\d+)\slayers.").findall(out)
                    assert len(matches) == 1
                    num = int(matches[0])
                    return num

                replace = False
                # replace = True
                drop = False

                # drop = True
                def gen_layer_files(file, dest):
                    results = []
                    num_layers = get_num_layers(file)
                    assert num_layers > 0
                    keep = None
                    if replace:
                        assert keep is not None and len(keep) == 1
                    for i in range(num_layers):
                        if keep and i not in keep:
                            continue
                        out_name = f"layer{i}"
                        out_file = Path(dest) / out_name
                        tflite_pack_args = [path, "-k", str(i), "--out", out_file]
                        utils.execute(self.pack_script, *tflite_pack_args)
                        assert out_file.is_file()
                        results.append(out_file)
                    return results

                layer_files = gen_layer_files(path, tmpdirname)

                for i, layer_file in enumerate(layer_files):
                    subrun = f"layer{i}"
                    layer_name = f"{name}_{subrun}"
                    layer_model = Model(layer_name, [layer_file])
                    ret = self.produce_artifacts(layer_model)
                    if not isinstance(ret, list):
                        ret = [ret]
                    assert len(ret) > 0, f"'{self.name}' frontend should produce at least one model"
                    max_outs = len(self.output_formats)
                    assert len(ret) <= max_outs, f"'{self.name}' frontend should not return more than {max_outs}"
                    if replace:
                        subrun = "default"
                    artifacts[subrun] = ret
                if drop:
                    del artifacts["default"]

            return artifacts, {}
        else:
            return super().generate(model)


class RelayFrontend(SimpleFrontend):
    FEATURES = Frontend.FEATURES | {"relayviz"}
    visualize_graph = cfg(False, cast=str2bool)
    relayviz_plotter = cfg("term")
    tvm_build_dir = required("tvm.build_dir")
    tvm_pythonpath = required("tvm.pythonpath")

    def __init__(self, features=None, config=None):
        super().__init__(
            "relay",
            ModelFormats.RELAY,
            features=features,
            config=config,
        )

    def produce_artifacts(self, model):
        assert len(self.input_formats) == len(model.paths) == 1
        artifacts = []

        name = model.name
        path = model.paths[0]
        ext = self.input_formats[0].extension
        with open(path, "rb") as handle:  # TODO: is an onnx model raw data or text?
            raw = handle.read()
            artifacts.append(Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"]))

        if not self.visualize_graph:
            assert len(self.output_formats) == 1
        else:
            assert len(self.output_formats) == 2

            def _relayviz(in_file, out_file, plotter_name, env={}):

                sys.path.append(env["PYTHONPATH"])
                os.environ["TVM_LIBRARY_PATH"] = env["TVM_LIBRARY_PATH"]
                try:
                    from tvm import parser
                    from tvm.contrib import relay_viz
                    from tvm.contrib.relay_viz.terminal import TermPlotter
                    from tvm.contrib.relay_viz.dot import DotPlotter
                except ImportError as ex:
                    logger.error("Missing Python package: tvm")
                    raise ex

                if plotter_name == "term":
                    plotter_cls = TermPlotter
                elif plotter_name == "dot":
                    plotter_cls = DotPlotter
                else:
                    raise RuntimeError(f"Invalid plotter name: {plotter_name}")

                with open(in_file, "r", encoding="utf-8") as relay_text:
                    text = relay_text.read()

                mod = parser.fromtext(text)

                plotter_inst = plotter_cls()
                viz = relay_viz.RelayVisualizer(mod, plotter=plotter_inst)
                out_file_base = os.path.splitext(out_file)[0]
                viz.render(filename=out_file_base)

            in_file = model.paths[0]
            ext = "" if self.relayviz_plotter == "term" else "pdf"
            with tempfile.TemporaryDirectory() as tmpdirname:
                out_file = str(Path(tmpdirname) / (f"relayviz.{ext}" if len(ext) > 0 else "relayviz"))
                proc = multiprocessing.Process(
                    target=_relayviz,
                    args=[in_file, out_file, self.relayviz_plotter],
                    kwargs={"env": {"PYTHONPATH": self.tvm_pythonpath, "TVM_LIBRARY_PATH": self.tvm_build_dir}},
                )
                proc.start()
                proc.join()

                if self.relayviz_plotter == "term":
                    with open(out_file, "r") as handle:
                        relayviz_text = handle.read()

                    relayviz_artifact = Artifact(
                        "relayviz.txt",
                        content=relayviz_text,
                        fmt=ArtifactFormat.TEXT,
                    )
                else:
                    with open(out_file, "rb") as handle:
                        relayviz_data = handle.read()

                    relayviz_artifact = Artifact(
                        f"relayviz.{ext}",
                        raw=relayviz_data,
                        fmt=ArtifactFormat.RAW,
                    )
                artifacts.append(relayviz_artifact)

        return artifacts


class PackedFrontend(Frontend):  # Inherit from TFLiteFrontend? -> how to do constructor?
    FEATURES = Frontend.FEATURES | {"packing", "packed"}
    ignore_existing = cfg(True, cast=str2bool)
    fake_pack = cfg(False, cast=str2bool)
    use_packed = cfg(True, cast=str2bool)
    check = cfg(False, cast=str2bool)
    packer_exe = required("packer.exe", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(name="packed", features=features, config=config)
        if self.fake_pack or self.ignore_existing:
            # assert self.use_packed
            self.input_formats = [ModelFormats.TFLITE]
        else:
            self.input_formats = [ModelFormats.PACKED, ModelFormats.TFLITE]

        # if self.use_packed:
        self.output_formats = [
            ModelFormats.PACKED,
            ModelFormats.TFLITE,
        ]  # Always copy over the input model as intermediate artifact for reproducability
        # TODO: add a Frontend.DEFAULT which can be used to turn off this behavoir (keep intermediate?)
        # else:
        #     self.output_formats = [ModelFormats.TFLITE]
        # Order of formats ir irrelevant here, hot for artifacts, the first one will always be the main object

    def produce_artifacts(self, model):
        tflite_data = None
        packed_data = None
        name = model.name

        if self.fake_pack or self.ignore_existing:  # -> ins: TFLITE
            # assert self.use_packed
            tflite_path = model.paths[0]

            with open(tflite_path, "rb") as handle:
                tflite_data = handle.read()
        else:  # -> ins: TFLITE, PACKED
            for path, fmt in zip(model.paths, model.formats):
                data_in = None
                with open(path, "rb") as handle:
                    data_in = handle.read()
                if fmt == ModelFormats.PACKED:
                    packed_data = data_in
                    break
                elif fmt == ModelFormats.TFLITE:
                    # tflite_data_in = data_in
                    break
                else:
                    raise RuntimeError(f"Unexpected model format: {fmt}")

        if packed_data is None:
            # Do packing
            with tempfile.TemporaryDirectory() as tmpdirname:
                logger.debug("Using temporary directory for packing results: %s", tmpdirname)
                packer_exe = self.packer_exe
                assert packer_exe is not None
                in_file = Path(tmpdirname) / "in.tflite"
                with open(in_file, "wb") as handle:
                    handle.write(tflite_data)
                out_file = Path(tmpdirname) / "out.tflm"
                utils.execute(packer_exe, in_file, out_file)
                with open(out_file, "rb") as handle:
                    packed_data = handle.read()

        if self.check:
            raise NotImplementedError

        tflite_artifact = Artifact(
            f"{name}.tflite",
            raw=tflite_data,
            fmt=ArtifactFormat.RAW,
            flags=["model"],
            optional=self.use_packed,
        )
        packed_artifact = Artifact(
            f"{name}.tflm",
            raw=packed_data,
            fmt=ArtifactFormat.RAW,
            optional=not self.use_packed,
        )

        if self.use_packed:
            return [packed_artifact, tflite_artifact]
        else:
            return [tflite_artifact, packed_artifact]


class ONNXFrontend(SimpleFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "onnx",
            ModelFormats.ONNX,
            features=features,
            config=config,
        )


class TorchFrontend(Frontend):
    def __init__(self, name: str, fmt: ModelFormats, features=None, config=None):
        super().__init__(
            name,
            input_formats=[fmt],
            output_formats=[fmt],
            features=features,
            config=config,
        )

    @property
    def supported_extensions(self):
        return [ext for fmt in self.input_formats for ext in fmt.extensions]

    def _make_hint(self, name, path, model_class=None, config=None):
        if path is None:
            assert model_class is not None
            paths = []
            formats = [ModelFormats.TORCH_PICKLE]
            classes = [model_class]
        else:
            paths = [path]
            ext = path.suffix[1:].lower() if path.suffix else ""
            assert ext in self.supported_extensions, f"Unsupported extension for torch model: {ext}"
            formats = [ModelFormats.from_extension(ext)]
            classes = []
        local_config = {} if config is None else config.copy()
        # model_name = model_class if model_class is not None else name
        model_name = name
        return Model(model_name, paths, config=local_config, alt=name, formats=formats, classes=classes)

    # def _path_contains_class(self, file_path, class_name):
    #     import ast

    #     with open(file_path, "r", encoding="utf-8") as handle:
    #         source = handle.read()
    #     tree = ast.parse(source, filename=str(file_path))
    #     return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))

    # def _find_class_in_dirs(self, class_name, directories):
    #     for directory in directories:
    #         if not directory.is_dir():
    #             continue
    #         for candidate in directory.glob("*.py"):
    #             if self._path_contains_class(candidate, class_name):
    #                 return candidate
    #     return None

    def _lookup_builtin_model(self, name):
        # import ast

        # builtins_file = Path(__file__).parent / "torch_models" / "quant_linear_test.py"
        # if not builtins_file.is_file():
        #     return None, None
        # with open(builtins_file, "r", encoding="utf-8") as handle:
        #     source = handle.read()
        # tree = ast.parse(source, filename=str(builtins_file))

        # alias_to_class = {}
        # for node in tree.body:
        #     if not isinstance(node, ast.Assign):
        #         continue
        #     if not any(isinstance(target, ast.Name) and target.id == "MODELS" for target in node.targets):
        #         continue
        #     if not isinstance(node.value, ast.Dict):
        #         continue
        #     for key_node, value_node in zip(node.value.keys, node.value.values):
        #         if isinstance(key_node, ast.Constant)
        #             and isinstance(key_node.value, str) and isinstance(value_node, ast.Name):
        #             alias_to_class[key_node.value] = value_node.id

        # if name in alias_to_class:
        #     return builtins_file, alias_to_class[name]
        from .torch_models import MODELS, MODEL_NAME_TO_MODEL, EagerModelFactory

        if name in MODELS:
            return None, MODELS[name]
        if name in MODEL_NAME_TO_MODEL:
            model, example_inputs, _, _ = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[name])
            model.example_input = example_inputs
            return None, model
        # if self._path_contains_class(builtins_file, name):
        #     return builtins_file, name
        return None, None

    def _get_search_directories(self, context=None):
        directories = []
        package_models = Path(__file__).parent / "torch_models"
        directories.append(package_models)
        if context is not None:
            model_paths = context.environment.paths.get("models", [])
            for model_path in model_paths:
                path = getattr(model_path, "path", model_path)
                directories.append(Path(path))
        return directories

    def lookup_models(self, names, config=None, context=None):
        hints = []
        config = config if config is not None else {}
        fmt = self.input_formats[0]
        for name in names:
            # path, model_class = self._lookup_builtin_model(name)
            _, model_class = self._lookup_builtin_model(name)
            if model_class is not None:
                assert fmt == ModelFormats.TORCH_PICKLE
                hints.append(self._make_hint(name, None, model_class=model_class, config=config))
                continue
            if fmt in [ModelFormats.TORCH_PYTHON, ModelFormats.TORCH_PICKLE, ModelFormats.TORCH_EXPORTED]:
                model_path = Path(name)
                suffix = model_path.suffix
                assert len(suffix) > 1
                ext = suffix[1:]
                assert model_path.is_file(), f"Not a file: {model_path}"
                assert ext in fmt.extensions, f"Unsupported extension: {ext}"
                name = model_path.stem
                from .torch_models.torch_utils import load_torch_model

                model, _, _ = load_torch_model(model_path)
                hints.append(self._make_hint(name, model_path, model_class=model, config=config))
                continue
            # if path is not None:
            #     hints.append(self._make_hint(name, Path(path), model_class=model_class, config=config))
            #     continue
            # class_file = self._find_class_in_dirs(name, self._get_search_directories(context=context))
            # if class_file is not None:
            #     hints.append(self._make_hint(name, class_file, model_class=name, config=config))
            #     continue
            raise RuntimeError(f"Could not find {self.name} model: {name}")
        return hints

    def produce_artifacts(self, model):
        artifacts = []
        if len(model.paths) == 0:
            assert len(model.classes) > 0
            assert len(model.classes) == 1
            model_class = model.classes[0]
            ext = "pkl"

            pickled_bytes = pickle.dumps(model_class)
            artifacts.append(
                Artifact(f"{model.name}.{ext}", raw=pickled_bytes, fmt=ArtifactFormat.RAW, flags=["model"])
            )
        else:
            path = model.paths[0]
            ext = model.formats[0].extension
            if ext == "py":
                # print("model", model, dir(model))
                # print("model.classes[0]", model.classes[0], dir(model.classes[0]))
                # input("!")
                with open(path, "r") as handle:
                    content = handle.read()
                artifacts.append(
                    Artifact(f"{model.name}.{ext}", content=content, fmt=ArtifactFormat.SOURCE, flags=["model"])
                )
            else:
                assert ext in ["pt", "pth", "pkl", "pickle"]
                with open(path, "rb") as handle:
                    raw = handle.read()
                artifacts.append(Artifact(f"{model.name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"]))
        return artifacts


class PTEFrontend(SimpleFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "pte",
            ModelFormats.PTE,
            features=features,
            config=config,
        )


class TorchPickleFrontend(TorchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "torch_pickle",
            fmt=ModelFormats.TORCH_PICKLE,
            features=features,
            config=config,
        )


class TorchPythonFrontend(TorchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "torch_python",
            fmt=ModelFormats.TORCH_PYTHON,
            features=features,
            config=config,
        )


class TorchExportedFrontend(TorchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "torch_exported",
            fmt=ModelFormats.TORCH_EXPORTED,
            features=features,
            config=config,
        )


class MLIRFrontend(SimpleFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "mlir",
            ModelFormats.MLIR,
            features=features,
            config=config,
        )


class PBFrontend(SimpleFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "pb",
            ModelFormats.PB,
            features=features,
            config=config,
        )


class PaddleFrontend(SimpleFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "paddle",
            ModelFormats.PADDLE,
            features=features,
            config=config,
        )


class BenchFrontend(SimpleFrontend):
    def __init__(self, name: str, prog_cls: Program, features=None, config=None):
        super().__init__(
            name,
            ModelFormats.NONE,
            features=features,
            config=config,
        )
        self.prog_cls = prog_cls

    def supported_names(self):
        raise NotImplementedError

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def lookup_models(self, names, config=None, context=None):
        ret = []
        for name in names:
            name = name.replace(f"{self.name}/", "")
            if name in self.supported_names:
                hint = self.prog_cls(
                    name,
                    alt=f"{self.name}/{name}",
                    config=config,
                )
                ret.append(hint)
        return ret

    def get_platform_config(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["template"] = self.name
        return ret


class SingleBenchFrontend(BenchFrontend):

    @property
    def supported_names(self):
        return [self.name]


class ExampleFrontend(BenchFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "example",
            ExampleProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # return ["hello_world", "foobar"]
        return [
            "hello_world",
            "foobar",
            "load_filter",
            "basic_hash",
            "temp_crypto",
        ]


class EmbenchFrontend(BenchFrontend):
    embench_src_dir = required("embench.src_dir", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "embench",
            EmbenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # TODO: automatic lookup
        return [
            "edn",
            "md5sum",
            "nettle-sha256",
            "nettle-aes",
            "ud",
            "matmult-int",
            "aha-mont64",
            "huffbench",
            "cubic",
            "nbody",
            "sglib-combined",
            "crc32",
            "wikisort",
            "slre",
            "qrduino",
            "minver",
            "picojpeg",
            "tarfind",
            "st",
            "nsichneu",
            "statemate",
            "primecount",
        ]

    def get_platform_defs(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["EMBENCH_DIR"] = self.embench_src_dir
        return ret


class EmbenchIoTFrontend(BenchFrontend):
    embench_iot_src_dir = required("embench_iot.src_dir", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "embench_iot",
            EmbenchIoTProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # TODO: automatic lookup
        return [
            # "cubic",
            # "nbody",
            # "minver",
            # "st",
            # "primecount", ???
            "aha-mont64",
            "crc32",
            "depthconv",
            "edn",
            "huffbench",
            "matmult-int",
            "md5sum",
            "nettle-aes",
            "nettle-sha256",
            "nsichneu",
            "picojpeg",
            "qrduino",
            "sglib-combined",
            "slre",
            "statemate",
            "tarfind",
            "ud",
            "wikisort",
            "xgboost",
        ]

    def get_platform_defs(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["EMBENCH_IOT_DIR"] = self.embench_iot_src_dir
        return ret


class EmbenchDSPFrontend(BenchFrontend):
    embench_dsp_src_dir = required("embench_dsp.src_dir", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "embench_dsp",
            EmbenchDSPProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # TODO: automatic lookup
        return [
            "biquad_cascade_df2T_f32_sos3_n1",
            "biquad_cascade_df2T_f32_sos3_n128",
            "dct4_2048_f32",
            "dct4_512_f32",
            "fir_f32_taps256_n1",
            "fir_f32_taps256_n128",
            "lms_f32_taps256_n1",
            "lms_f32_taps256_n128",
            "rfft2048_f32",
            "rfft512_f32",
        ]

    def get_platform_defs(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["EMBENCH_DSP_DIR"] = self.embench_dsp_src_dir
        return ret


class TaclebenchFrontend(BenchFrontend):
    taclebench_src_dir = required("taclebench.src_dir", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "taclebench",
            TaclebenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # TODO: automatic lookup
        return [
            "test/test3",
            "test/cover",
            "test/duff",
            "app/powerwindow",
            "app/lift",
            "kernel/deg2rad",
            "kernel/matrix1",
            "kernel/binarysearch",
            "kernel/pm",
            "kernel/sha",
            "kernel/filterbank",
            "kernel/md5",
            "kernel/fir2dim",
            "kernel/fft",
            "kernel/minver",
            "kernel/lms",
            "kernel/bitcount",
            "kernel/st",
            "kernel/bsort",
            "kernel/bitonic",
            "kernel/iir",
            "kernel/prime",
            "kernel/jfdctint",
            "kernel/recursion",
            "kernel/complex_updates",
            "kernel/cosf",
            "kernel/insertsort",
            "kernel/fac",
            "kernel/rad2deg",
            "kernel/isqrt",
            "kernel/cubic",
            "kernel/ludcmp",
            "kernel/quicksort",
            "kernel/countnegative",
            "sequential/epic",
            "sequential/huff_dec",
            "sequential/fmref",
            "sequential/h264_dec",
            "sequential/dijkstra",
            "sequential/adpcm_dec",
            "sequential/adpcm_enc",
            "sequential/gsm_dec",
            "sequential/rijndael_dec",
            "sequential/g723_enc",
            "sequential/huff_enc",
            "sequential/statemate",
            "sequential/susan",
            "sequential/gsm_enc",
            "sequential/ndes",
            "sequential/audiobeam",
            "sequential/rijndael_enc",
            "sequential/cjpeg_transupp",
            "sequential/ammunition",
            "sequential/mpeg2",
            "sequential/anagram",
            "sequential/cjpeg_wrbmp",
            "sequential/petrinet",
        ]

    def get_platform_defs(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["TACLEBENCH_DIR"] = self.taclebench_src_dir
        return ret


class PolybenchFrontend(BenchFrontend):
    dataset = cfg("large")
    polybench_src_dir = required("polybench.src_dir", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "polybench",
            PolybenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # TODO: automatic lookup
        return [
            "linear-algebra/solvers/gramschmidt",
            "linear-algebra/solvers/ludcmp",
            "linear-algebra/solvers/trisolv",
            "linear-algebra/solvers/durbin",
            "linear-algebra/solvers/lu",
            "linear-algebra/solvers/cholesky",
            "linear-algebra/kernels/atax",
            "linear-algebra/kernels/3mm",
            "linear-algebra/kernels/mvt",
            "linear-algebra/kernels/2mm",
            "linear-algebra/kernels/bicg",
            "linear-algebra/kernels/doitgen",
            "linear-algebra/blas/trmm",
            "linear-algebra/blas/gemver",
            "linear-algebra/blas/syrk",
            "linear-algebra/blas/gesummv",
            "linear-algebra/blas/syr2k",
            "linear-algebra/blas/symm",
            "linear-algebra/blas/gemm",
            "stencils/fdtd-2d",
            "stencils/seidel-2d",
            "stencils/adi",
            "stencils/jacobi-1d",
            "stencils/jacobi-2d",
            "stencils/heat-3d",
            "datamining/covariance",
            "datamining/correlation",
            "medley/deriche",
            "medley/nussinov",
            "medley/floyd-warshall",
        ]

    def get_platform_defs(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["POLYBENCH_DIR"] = self.polybench_src_dir
            ret["POLYBENCH_DATASET"] = self.dataset.upper() + "_DATASET"
        return ret


class CoremarkFrontend(SingleBenchFrontend):
    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(
            "coremark",
            CoremarkProgram,
            features=features,
            config=config,
        )


class DhrystoneFrontend(SingleBenchFrontend):
    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(
            "dhrystone",
            DhrystoneProgram,
            features=features,
            config=config,
        )


class MathisFrontend(BenchFrontend):
    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(
            "mathis",
            MathisProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "to_upper",
            "add8",
            "add16",
            "gather_add8",
            "gather_add16",
            "scatter_add8",
            "scatter_add16",
            "dot8",
            "dot16",
            "saxpy8",
            "saxpy16",
            "matmul8",
            "matmul16",
            "matmul8_a",
            "matmul16_a",
            "transposed_matmul8",
            "transposed_matmul16",
            "transposed_matmul8_a",
            "transposed_matmul16_a",
            "transposed_matmul8_b",
            "transposed_matmul16_b",
        ]


class MibenchFrontend(BenchFrontend):
    mibench_src_dir = required("mibench.src_dir", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "mibench",
            MibenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        # TODO: automatic lookup
        return [
            "telecomm/FFT",
            "telecomm/CRC32",
            "automotive/susan",
            "automotive/basicmath_small",
            "automotive/basicmath_large",
            "automotive/bitcount_small",
            "automotive/bitcount_large",
            "automotive/qsort_small",
            "automotive/qsort_large",
            "security/sha",
            "security/rijndael",
            "network/dijkstra",
            "office/stringsearch",
        ]

    def get_platform_defs(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["MIBENCH_DIR"] = self.mibench_src_dir

        return ret


class LayerGenFrontend(Frontend):
    """Example input:
    1,1024 1 fully_connected ,10,relu
    1,1024 1 fully_connected ,100,relu
    1,1024 1 fully_connected ,2,relu
    1,1024 1 fully_connected ,4,relu
    1,1024 1 fully_connected ,8,relu
    1,1024 1 fully_connected ,16,relu
    1,1024 1 fully_connected ,32,relu
    1,1024 1 fully_connected ,64,relu
    1,1024 1 fully_connected ,128,relu
    1,1024 1 fully_connected ,256,relu
    1,1024 1 fully_connected ,512,relu
    """

    FEATURES = Frontend.FEATURES

    DEFAULTS = {"fmt": "tflite"}  # TODO: relay
    layergen_exe = required("layergen.exe", cast=Path)

    def __init__(self, features=None, config=None):
        super().__init__(
            "layergen",
            input_formats=[ModelFormats.TEXT],
            output_formats=[ModelFormats.TFLITE, ModelFormats.RELAY],
            features=features,
            config=config,
        )

    @property
    def fmt(self):
        value = self.config["fmt"]
        value = value.upper()
        assert value in ["TFLITE", "RELAY"]
        return value

    def produce_artifacts(self, model):
        pass

    # def produce_artifacts(self, model):
    #     artifacts = {}
    #     name = model.name
    #     path = model.paths[0]
    #     ext = ModelFormats[self.fmt].extension
    #     print("ext", ext)
    #     with open(path, "r") as handle:
    #         content = handle.read()
    #     lines = content.strip().split("\n")
    #     print("lines", lines, list(filter(None, lines)))
    #     assert len(lines) > 0, "Empty file not allowed."

    #     def helper(args):
    #         args = args.split(" ")
    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             out = Path(tmpdirname) / f"out.{ext}"
    #             utils.python(self.layergen_exe, self.fmt.lower(), out, *args, cwd=tmpdirname)
    #             # TODO: log output
    #             with open(out, "rb") as handle:
    #                 raw = handle.read()
    #             return raw

    #     if len(lines) > 1:
    #         for i, args in enumerate(lines):
    #             name = f"model{i}"
    #             raw = helper(args)
    #             artifact = Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"])
    #         artifacts[name] = [artifact]
    #     else:
    #         artifacts["default"] = []
    #     return artifacts

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = {}
        name = model.name
        path = model.paths[0]
        ext = ModelFormats[self.fmt].extension
        with open(path, "r") as handle:
            content = handle.read()
        lines = content.strip().split("\n")
        assert len(lines) > 0, "Empty file not allowed."

        def helper(args):
            args = args.split(" ")
            with tempfile.TemporaryDirectory() as tmpdirname:
                out = Path(tmpdirname) / f"out.{ext}"
                utils.python(self.layergen_exe, self.fmt.lower(), out, *args, cwd=tmpdirname)
                # TODO: log output
                with open(out, "rb") as handle:
                    raw = handle.read()
                return raw

        if len(lines) > 1:
            for i, args in enumerate(lines):
                name = f"model{i}"
                raw = helper(args)
                artifact = Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"])
                artifacts[name] = [artifact]
            # TODO: fix this
            artifacts["default"] = artifacts["model0"]  # Dummy model because default artifacts can not be empty
        else:
            name = "default"
            raw = helper(lines[0])
            artifact = Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW, flags=["model"])
            artifacts[name] = [artifact]
        return artifacts, {}


class OpenASIPFrontend(BenchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "openasip",
            OpenASIPProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "sha256",
            "aes",
            "crc",
        ]


class RVVBenchFrontend(BenchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "rvv_bench",
            RVVBenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "instructions_scalar",
            "instructions_rvv",
            "memcpy",
            "memset",
            "utf8_count",
            "strlen",
            "mergelines",
            "mandelbrot",
            "chacha20",
            "poly1305",
            "ascii_to_utf16",
            "ascii_to_uft32",
            "byteswap",
            "LUT4",
            "LUT6",
            "hist",
            "base64_encode",
        ]


class ISSBenchFrontend(BenchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "iss_bench",
            ISSBenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "large_block",
            "branch_heavy",
            "mem_heavy",
        ]


class CryptoBenchFrontend(BenchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "crypto_bench",
            CryptoBenchProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "mceliece348864_encrypt",
            "mceliece348864_encrypt_rvv",
            "mceliece348864_decrypt",
            "mceliece348864_decrypt_rvv",
            "mceliece348864_keypair",
            "mceliece348864_keypair_rvv",
            "hqc-128",
            "hqc-192",
            "hqc-256",
        ]


class CmsisDSPFrontend(BenchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "cmsis_dsp",
            CmsisDSPProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "arm_abs_q15",
            "arm_abs_q31",
        ]

    def get_platform_config(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["template"] = "cmsis_dsp_bench"
        return ret


class CmsisNNFrontend(BenchFrontend):

    def __init__(self, features=None, config=None):
        super().__init__(
            "cmsis_nn",
            CmsisNNProgram,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "arm_nn_activation_s16_tanh",
            "arm_nn_activation_s16_sigmoid",
        ]

    def get_platform_config(self, platform):
        ret = {}
        if platform in ["mlif", "mlif_litex"]:
            ret["template"] = "cmsis_nn_bench"
        return ret
