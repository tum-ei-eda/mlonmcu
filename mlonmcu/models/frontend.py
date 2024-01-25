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
import re
import time
import tempfile
import multiprocessing
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np

from mlonmcu.feature.features import get_matching_features
from mlonmcu.models.model import (
    ModelFormats,
    Model,
    ExampleProgram,
    EmbenchProgram,
    TaclebenchProgram,
    PolybenchProgram,
    CoremarkProgram,
    DhrystoneProgram,
    MathisProgram,
    MibenchProgram,
)
from mlonmcu.models.lookup import lookup_models
from mlonmcu.feature.type import FeatureType
from mlonmcu.config import filter_config, str2bool
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.setup import utils
from mlonmcu.target.metrics import Metrics

from mlonmcu.logging import get_logger


logger = get_logger()


class Frontend(ABC):
    FEATURES = {"validate"}

    DEFAULTS = {
        "use_inout_data": False,
        # the following should be configured using gen_data feature
        "gen_data": False,
        "gen_data_fill_mode": None,
        "gen_data_file": None,
        "gen_data_number": None,
        "gen_data_fmt": None,
        # the following should be configured using gen_ref_data feature
        "gen_ref_data": False,
        "gen_ref_data_mode": None,
        "gen_ref_data_file": None,
        "gen_ref_data_fmt": None,
    }

    REQUIRED = set()
    OPTIONAL = set()

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

    @property
    def use_inout_data(self):
        value = self.config["use_inout_data"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gen_data(self):
        value = self.config["gen_data"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gen_data_fill_mode(self):
        value = self.config["gen_data_fill_mode"]
        assert value in ["random", "ones", "zeros", "file", "dataset"]
        return value

    @property
    def gen_data_file(self):
        return self.config["gen_data_file"]

    @property
    def gen_data_number(self):
        return int(self.config["gen_data_number"])

    @property
    def gen_data_fmt(self):
        value = self.config["gen_data_fmt"]
        assert value in ["npy", "npz"]
        return value

    @property
    def gen_ref_data(self):
        value = self.config["gen_data"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gen_ref_data_mode(self):
        value = self.config["gen_ref_data_mode"]
        assert value in ["file", "model"]
        return value

    @property
    def gen_ref_data_file(self):
        return self.config["gen_ref_data_file"]

    @property
    def gen_ref_data_fmt(self):
        value = self.config["gen_ref_data_fmt"]
        assert value in ["npy", "npz"]
        return value

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
        model_dir = Path(model.paths[0]).parent.resolve()
        metadata = model.metadata
        in_paths = []
        out_paths = []
        input_shapes = {}
        output_shapes = {}
        input_types = {}
        output_types = {}
        input_quant_details = {}
        output_quant_details = {}
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
                quantize = inp.get("quantize", None)
                if name and shape:
                    input_shapes[name] = shape
                if name and ty:
                    input_types[name] = ty
                if name and quantize:
                    quant_scale = quantize.get("scale", None)
                    quant_zero_shift = quantize.get("zero_shift", None)
                    quant_dtype = quantize.get("dtype", None)
                    quant_details = [quant_scale, quant_zero_shift, quant_dtype]
                    input_quant_details[name] = quant_details
                if self.use_inout_data or (self.gen_data and self.gen_data_fill_mode == "file" and self.gen_data_file == "auto"):
                    if "example_input" in inp and "path" in inp["example_input"]:
                        in_data_dir = Path(inp["example_input"]["path"])
                        # TODO: this will only work with relative paths to model dir! (Fallback to parent directories?)
                        in_path = model_dir / in_data_dir
                        assert (
                            in_path.is_dir()
                        ), f"Input data directory defined in model metadata does not exist: {in_path}"
                        in_paths.append(in_path)
            assert "output_nodes" in network
            outs = network["output_nodes"]
            for outp in outs:
                name = outp.get("name", None)
                shape = outp.get("shape", None)
                ty = outp.get("dtype", None)
                if ty is None:
                    ty = outp.get("type", None)  # legacy
                dequantize = outp.get("dequantize", None)
                if name and shape:
                    output_shapes[name] = shape
                if name and ty:
                    output_types[name] = ty
                if name and dequantize:
                    quant_scale = dequantize.get("scale", None)
                    quant_zero_shift = dequantize.get("zero_shift", None)
                    quant_dtype = dequantize.get("dtype", None)
                    quant_details = [quant_scale, quant_zero_shift, quant_dtype]
                    output_quant_details[name] = quant_details
                if self.use_inout_data or (self.gen_ref_data and self.gen_ref_data_mode == "file" and self.gen_ref_data_file == "auto"):
                    if "test_output_path" in outp:
                        out_data_dir = Path(outp["test_output_path"])
                        out_path = model_dir / out_data_dir
                        assert (
                            out_path.is_dir()
                        ), f"Output data directory defined in model metadata does not exist: {out_path}"
                        out_paths.append(out_path)
        else:
            fallback_in_path = model_dir / "input"
            if fallback_in_path.is_dir():
                in_paths.append(fallback_in_path)
            fallback_out_path = model_dir / "output"
            if fallback_out_path.is_dir():
                out_paths.append(fallback_out_path)

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
                input_names, input_shapes, input_types, input_quant_details, output_names, output_shapes, output_types, output_quant_details = self.extract_model_info(model)
            except NotImplementedError:
                logger.warning("Model info could not be extracted.")

        # Detect model support code (Allow overwrite in metadata YAML)
        support_path = model_dir / "support"
        if support_path.is_dir():
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
        model_info_dict = {
            "input_names": input_names,
            "output_names": output_names,
            "input_shapes": list(input_shapes.values()),
            "output_shapes": list(output_shapes.values()),
            "input_types": list(input_types.values()),
            "output_types": list(output_types.values()),
            "input_quant_details": list(input_quant_details.values()),
            "output_quant_details": list(output_quant_details.values()),
        }
        print("model_info_dict", model_info_dict)
        artifacts = []
        if self.gen_data:
            inputs_data = []
            if self.gen_data_fill_mode in ["zeros", "ones", "random"]:
                for i in range(self.gen_data_number):
                    print("i", i)
                    data = {}
                    for ii, input_name in enumerate(input_names):
                        print("ii", ii)
                        assert input_name in input_types, f"Unknown dtype for input: {input_name}"
                        dtype = input_types[input_name]
                        quant = input_quant_details.get(input_name, None)
                        if quant:
                            _, _, ty = quant
                            # dtype = ty
                        assert input_name in input_shapes, f"Unknown shape for input: {input_name}"
                        shape = input_shapes[input_name]
                        if self.gen_data_fill_mode == "zeros":
                            arr = np.zeros(shape, dtype=dtype)
                        elif self.gen_data_fill_mode == "ones":
                            arr = np.ones(shape, dtype=dtype)
                        elif self.gen_data_fill_mode == "random":
                            if "float" in dtype:
                                arr = np.random.rand(*shape).astype(dtype)
                            elif "int" in dtype:
                                arr = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, size=shape, dtype=dtype)
                            else:
                                assert False
                        else:
                            assert False
                        data[input_name] = arr
                    assert len(data) > 0
                    inputs_data.append(data)
            elif self.gen_data_fill_mode == "file":
                if self.gen_data_file == "auto":
                    len(in_paths) > 0
                    print("in_paths", in_paths)
                    if len(in_paths) == 1:
                        if in_paths[0].is_dir():
                            files = list(in_paths[0].iterdir())
                    else:
                        files = in_paths
                    temp = {}
                    for file in files:
                        if not isinstance(file, Path):
                            file = Path(file)
                        assert file.is_file()
                        print("file", file)
                        basename, ext = file.stem, file.suffix
                        if ext == ".bin":
                            if "_" in basename:
                                i, ii = basename.split("_", 1)
                                i = int(i)
                                ii = int(ii)
                            else:
                                i = int(basename)
                                ii = 0
                            with open(file, "rb") as f:
                                data = f.read()
                            if i not in temp:
                                temp[i] = {}
                            temp[i][ii] = data
                        elif ext in [".npy", ".npz"]:
                            raise NotImplementedError
                        else:
                            raise RuntimeError(f"Unsupported ext: {ext}")
                    # print("temp", temp)
                    assert len(temp) > 0
                    for i in range(min(self.gen_data_number, len(temp))):
                        print("i", i)
                        assert i in temp
                        data = {}
                        for ii, input_name in enumerate(input_names):
                            print("ii", ii)
                            assert ii in temp[i]
                            assert input_name in input_types, f"Unknown dtype for input: {input_name}"
                            dtype = input_types[input_name]
                            quant = input_quant_details.get(input_name, None)
                            if quant:
                                _, _, ty = quant
                                dtype = ty
                            arr = np.frombuffer(temp[i][ii], dtype=dtype)
                            assert input_name in input_shapes, f"Unknown shape for input: {input_name}"
                            shape = input_shapes[input_name]
                            arr = np.reshape(arr, shape)
                            data[input_name] = arr
                        inputs_data.append(data)
                else:
                    assert self.gen_data_file is not None, "Missing value for gen_data_file"
                    file = Path(self.gen_data_file)
                    assert file.is_file(), f"File not found: {file}"
                # for i, input_name in enumerate(input_names):

            elif self.gen_data_fill_mode == "dataset":
                raise NotImplementedError
            else:
                raise RuntimeError(f"unsupported fill_mode: {self.gen_data_fill_mode}")
            print("inputs_data", inputs_data)
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
            outputs_data = []
            if self.gen_ref_data_mode == "model":
                assert len(inputs_data) > 0
                for i, input_data in enumerate(inputs_data):
                    print("i", i)
                    print("model", model, type(model))
                    # input("321?")
                    output_data = self.inference(model, input_data, quant=False, dequant=True)
                    print("output_data", output_data)
                    outputs_data.append(output_data)
                    # input("321!")

            elif self.gen_ref_data_mode == "file":
                if self.gen_ref_data_file == "auto":
                    len(out_paths) > 0
                    if len(out_paths) == 1:
                        if out_paths[0].is_dir():
                            files = list(out_paths[0].iterdir())
                    else:
                        files = out_paths
                    temp = {}
                    for file in files:
                        if not isinstance(file, Path):
                            file = Path(file)
                        assert file.is_file()
                        print("file", file)
                        basename, ext = file.stem, file.suffix
                        if ext == ".bin":
                            if "_" in basename:
                                i, ii = basename.split("_", 1)
                                i = int(i)
                                ii = int(ii)
                            else:
                                i = int(basename)
                                ii = 0
                            with open(file, "rb") as f:
                                data = f.read()
                            if i not in temp:
                                temp[i] = {}
                            temp[i][ii] = data
                        elif ext in [".npy", ".npz"]:
                            raise NotImplementedError
                        else:
                            raise RuntimeError(f"Unsupported ext: {ext}")
                    # print("temp", temp)
                    # TODO: handle case where there are more output samples than input samples?
                    for i in range(len(temp)):
                        print("i", i)
                        assert i in temp
                        data = {}
                        for ii, output_name in enumerate(output_names):
                            print("ii", ii)
                            assert ii in temp[i]
                            assert output_name in output_types, f"Unknown dtype for output: {output_name}"
                            dtype = output_types[output_name]
                            dequant = output_quant_details.get(output_name, None)
                            if dequant:
                                _, _, ty = dequant
                                dtype = ty
                            arr = np.frombuffer(temp[i][ii], dtype=dtype)
                            assert output_name in output_shapes, f"Unknown shape for output: {output_name}"
                            shape = output_shapes[output_name]
                            arr = np.reshape(arr, shape)
                            data[output_name] = arr
                        outputs_data.append(data)
                else:
                    assert self.gen_data_file is not None, "Missing value for gen_data_file"
                    file = Path(self.gen_data_file)
                    assert file.is_file(), f"File not found: {file}"
            else:
                raise RuntimeError(f"unsupported fill_mode: {self.gen_ref_data_mode}")
            print("outputs_data", outputs_data)
            fmt = self.gen_data_fmt
            if fmt == "npy":
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tempfilename = Path(tmpdirname) / "outputs_ref.npy"
                    np.save(tempfilename, outputs_data)
                    with open(tempfilename, "rb") as f:
                        raw = f.read()
            elif fmt == "npz":
                raise NotImplementedError
            else:
                raise RuntimeError(f"Unsupported fmt: {fmt}")
            assert raw
            outputs_data_artifact = Artifact(f"outputs_ref.{fmt}", raw=raw, fmt=ArtifactFormat.BIN, flags=("outputs_ref", fmt))
            artifacts.append(outputs_data_artifact)
        # nested version
        # model_info_dict = {
        #     "inputs": [
        #         {
        #             "name": "input_1",
        #             "shape": [1, 1014],
        #             "type": "int8",
        #         }
        #     ],
        #     "outputs": [
        #         {
        #             "name": "output",
        #             "shape": [1, 10],
        #             "type": "int8",
        #         }
        #     ],
        # }
        import yaml
        content = yaml.dump(model_info_dict)
        model_info_artifact = Artifact("model_info.yml", content=content, fmt=ArtifactFormat.TEXT, flags=("model_info",))
        artifacts.append(model_info_artifact)
        return artifacts

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = []

        count = len(model.paths)
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


# TODO: move to frontends.py
# TODO: frontend parsed metadata instead of lookup.py?
# TODO: how to find inout_data?
class TfLiteFrontend(SimpleFrontend):
    FEATURES = Frontend.FEATURES | {"visualize", "split_layers", "tflite_analyze", "gen_data", "gen_ref_data"}

    DEFAULTS = {
        **Frontend.DEFAULTS,
        "visualize_enable": False,
        "visualize_script": None,
        "split_layers": False,
        "pack_script": None,
        "analyze_enable": False,
        "analyze_script": None,
    }

    REQUIRED = Frontend.REQUIRED

    OPTIONAL = SimpleFrontend.OPTIONAL | {"tflite_analyze.script"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "tflite",
            ModelFormats.TFLITE,
            features=features,
            config=config,
        )

    @property
    def visualize_enable(self):
        value = self.config["visualize_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def split_layers(self):
        value = self.config["split_layers"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def visualize_script(self):
        return self.config["visualize_script"]

    @property
    def pack_script(self):
        return self.config["pack_script"]

    @property
    def analyze_enable(self):
        value = self.config["analyze_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def analyze_script(self):
        return self.config["analyze_script"]

    def extract_model_info(self, model: Model):
        import tensorflow as tf
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
            if "quantization" in inp:
                scale, zero_point = inp["quantization"]
                quant = [scale, zero_point, "float32"]
                input_quant_details[name] = quant
        for outp in output_details:
            name = str(outp["name"])
            output_names.append(name)
            output_shapes[name] = outp["shape"].tolist()
            output_types[name] = np.dtype(outp["dtype"]).name
            if "quantization" in outp:
                scale, zero_point = outp["quantization"]
                quant = [scale, zero_point, "float32"]
                output_quant_details[name] = quant
        return input_names, input_shapes, input_types, input_quant_details, output_names, output_shapes, output_types, output_quant_details

    def inference(self, model: Model, input_data: Dict[str, np.array], quant=False, dequant=False):
        import tensorflow as tf
        model_path = str(model.paths[0])
        interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
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
            input_scale, input_zero_point = input_details[0]['quantization']
            print("Input scale:", input_scale)
            print("Input zero point:", input_zero_point)
            print()
            np_features = (np_features / input_scale) + input_zero_point
            np_features = np.around(np_features)
        np_features = np_features.astype(input_type)
        np_features = np_features.reshape(input_shape)
        print("np_features", np_features)
        interpreter.set_tensor(input_details[0]['index'], np_features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # If the output type is int8 (quantized model), rescale data
        assert len(output_details) == 1, "Multi-outputs not yet supported"
        output_type = output_details[0]["dtype"]
        output_name = output_details[0]["name"]
        if dequant and output_type == np.int8:
            output_scale, output_zero_point = output_details[0]["quantization"]
            print("Raw output scores:", output)
            print("Output scale:", output_scale)
            print("Output zero point:", output_zero_point)
            print()
            output = output_scale * (output.astype(np.float32) - output_zero_point)

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
                utils.python(self.analyze_script, *args, print_output=False)

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

                utils.python(self.visualize_script, in_file, out_file, print_output=False)

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
                    out = utils.exec_getout(self.pack_script, *tflite_pack_args, print_output=False)
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
                        utils.exec_getout(self.pack_script, *tflite_pack_args, print_output=False)
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

    DEFAULTS = {**Frontend.DEFAULTS, "visualize_graph": False, "relayviz_plotter": "term"}

    REQUIRED = Frontend.REQUIRED | {"tvm.build_dir", "tvm.pythonpath"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "relay",
            ModelFormats.RELAY,
            features=features,
            config=config,
        )

    @property
    def visualize_graph(self):
        value = self.config["visualize_graph"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def relayviz_plotter(self):
        return self.config["relayviz_plotter"]

    @property
    def tvm_build_dir(self):
        return self.config["tvm.build_dir"]

    @property
    def tvm_pythonpath(self):
        return self.config["tvm.pythonpath"]

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
                import sys
                import os

                sys.path.append(env["PYTHONPATH"])
                os.environ["TVM_LIBRARY_PATH"] = env["TVM_LIBRARY_PATH"]
                from tvm import parser
                from tvm.contrib import relay_viz
                from tvm.contrib.relay_viz.terminal import TermPlotter
                from tvm.contrib.relay_viz.dot import DotPlotter

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

    DEFAULTS = {
        **Frontend.DEFAULTS,
        "ignore_existing": True,
        "fake_pack": False,  # Pretend that every compatible tensor is packable
        # (best case scenerio, TODO: rename to force_pack?)
        "use_packed": True,
        "check": False,  # Unimplemented
    }

    REQUIRED = {"packer.exe"}  # TODO move to feature?

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

    @property
    def ignore_existing(self):
        value = self.config["ignore_existing"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def fake_pack(self):
        value = self.config["fake_pack"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def use_packed(self):
        value = self.config["use_packed"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def check(self):
        value = self.config["check"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
                packer_exe = self.config["packer_exe"]
                assert packer_exe is not None
                in_file = Path(tmpdirname) / "in.tflite"
                with open(in_file, "wb") as handle:
                    handle.write(tflite_data)
                out_file = Path(tmpdirname) / "out.tflm"
                utils.exec_getout(packer_exe, in_file, out_file)
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
            flags=["model"],
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


class ExampleFrontend(SimpleFrontend):
    def __init__(self, features=None, config=None):
        super().__init__(
            "example",
            ModelFormats.NONE,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return ["hello_world", "foobar"]

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("example/", "")
            if name in self.supported_names:
                hint = ExampleProgram(
                    name,
                    alt=f"example/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "example"
        return ret


class EmbenchFrontend(SimpleFrontend):
    REQUIRED = {"embench.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "embench",
            ModelFormats.NONE,
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

    # @property
    # def skip_backend(self):
    #     return True

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("embench/", "")
            if name in self.supported_names:
                hint = EmbenchProgram(
                    name,
                    alt=f"embench/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["EMBENCH_DIR"] = Path(self.config["embench.src_dir"])
        return ret

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "embench"
        return ret


class TaclebenchFrontend(SimpleFrontend):
    REQUIRED = {"taclebench.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "taclebench",
            ModelFormats.NONE,
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

    # @property
    # def skip_backend(self):
    #     return True

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("taclebench/", "")
            if name in self.supported_names:
                hint = TaclebenchProgram(
                    name,
                    alt=f"taclebench/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["TACLEBENCH_DIR"] = Path(self.config["taclebench.src_dir"])
        return ret

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "taclebench"
        return ret


class PolybenchFrontend(SimpleFrontend):
    REQUIRED = {"polybench.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "polybench",
            ModelFormats.NONE,
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

    # @property
    # def skip_backend(self):
    #     return True

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("polybench/", "")
            if name in self.supported_names:
                hint = PolybenchProgram(
                    name,
                    alt=f"polybench/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["POLYBENCH_DIR"] = Path(self.config["polybench.src_dir"])
        return ret

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "polybench"
        return ret


class CoremarkFrontend(SimpleFrontend):
    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(
            "coremark",
            ModelFormats.NONE,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "coremark",
        ]

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("coremark/", "")
            if name in self.supported_names:
                hint = CoremarkProgram(
                    name,
                    alt=f"coremark/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "coremark"
        return ret


class DhrystoneFrontend(SimpleFrontend):
    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(
            "dhrystone",
            ModelFormats.NONE,
            features=features,
            config=config,
        )

    @property
    def supported_names(self):
        return [
            "dhrystone",
        ]

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("dhrystone/", "")
            if name in self.supported_names:
                hint = DhrystoneProgram(
                    name,
                    alt=f"dhrystone/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "dhrystone"
        return ret


class MathisFrontend(SimpleFrontend):
    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(
            "mathis",
            ModelFormats.NONE,
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

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("mathis/", "")
            if name in self.supported_names:
                hint = MathisProgram(
                    name,
                    alt=f"mathis/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "mathis"
        return ret


class MibenchFrontend(SimpleFrontend):
    REQUIRED = {"mibench.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "mibench",
            ModelFormats.NONE,
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
            "automotive/basicmath",
            "automotive/bitcount",
            "automotive/qsort",
            "security/sha",
            "security/rijndael",
            "network/dijkstra",
            "office/stringsearch",
        ]

    # @property
    # def skip_backend(self):
    #     return True

    def lookup_models(self, names, context=None):
        ret = []
        for name in names:
            name = name.replace("mibench/", "")
            if name in self.supported_names:
                hint = MibenchProgram(
                    name,
                    alt=f"mibench/{name}",
                )
                ret.append(hint)
        return ret

    def generate(self, model) -> Tuple[dict, dict]:
        artifacts = [Artifact("dummy_model", raw=bytes(), fmt=ArtifactFormat.RAW, flags=["model", "dummy"])]

        return {"default": artifacts}, {}

    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["MIBENCH_DIR"] = Path(self.config["mibench.src_dir"])

        return ret

    def get_platform_config(self, platform):
        ret = {}
        if platform == "mlif":
            ret["template"] = "mibench"

        return ret


class LayerGenFrontend(Frontend):
    FEATURES = Frontend.FEATURES

    DEFAULTS = {
        **Frontend.DEFAULTS,
        "fmt": "tflite",  # TODO: relay
    }

    REQUIRED = Frontend.REQUIRED | {"layergen.exe"}

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

    @property
    def layergen_exe(self):
        return Path(self.config["layergen.exe"])

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
    #             utils.python(self.layergen_exe, self.fmt.lower(), out, *args, print_output=True, cwd=tmpdirname)
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
                utils.python(self.layergen_exe, self.fmt.lower(), out, *args, print_output=True, cwd=tmpdirname)
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
