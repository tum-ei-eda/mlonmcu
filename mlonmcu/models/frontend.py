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
import tempfile
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Callable
import logging
import copy

from mlonmcu.feature.features import get_matching_features
from mlonmcu.models.model import Model, ModelFormats
from mlonmcu.feature.feature import Feature
from mlonmcu.feature.type import FeatureType
from mlonmcu.config import filter_config
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.logging import get_logger

from .utils import get_data_source

logger = get_logger()


class Frontend(ABC):

    FEATURES = ["validate"]

    DEFAULTS = {
        "use_inout_data": False,
    }

    REQUIRED = []

    def __init__(self, name, input_formats=None, output_formats=None, features=None, config=None):
        self.name = name
        self.input_formats = input_formats if input_formats else []
        self.output_formats = output_formats if output_formats else []
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)

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
        return bool(self.config["use_inout_data"])

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

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.FRONTEND)
        for feature in features:
            assert (  # If this assertion occurs, continue with the next frontend instea dof failing (TODO: create custom exception type)
                feature.name in self.FEATURES
            ), f"Incompatible feature: {feature.name}"
            # Instead we might introduce self.compatible and set it to true at this line
            feature.add_frontend_config(self.name, self.config)
        return features

    @abstractmethod
    # def produce_artifacts(self, model):
    def produce_artifacts(self, model):
        pass

    def process_metadata(self, model, cfg=None):
        model_dir = Path(model.paths[0]).parent
        metadata = model.metadata
        if self.use_inout_data:
            in_paths = []
            out_paths = []
            if metadata is not None and "network_parameters" in metadata:
                network = metadata["network_parameters"]
                assert "input_nodes" in network
                ins = network["input_nodes"]
                for inp in ins:
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
                    if "test_output_path" in outp:
                        out_data_dir = Path(outp["test_output_path"])
                        out_path = model_dir / out_data_dir
                        assert (
                            in_path.is_dir()
                        ), f"Output data directory defined in model metadata does not exist: {out_path}"
                        out_paths.append(out_path)
            else:
                fallback_in_path = model_dir / "input"
                if fallback_in_path.is_dir():
                    in_paths.append(fallback_in_path)
                fallback_out_path = model_dir / "output"
                if fallback_out_path.is_dir():
                    out_paths.append(fallback_out_path)
            data_src = get_data_source(in_paths, out_paths)
            data_artifact = Artifact("data.c", content=data_src, fmt=ArtifactFormat.SOURCE)
        else:
            data_artifact = None

        if metadata is not None and "backends" in metadata:
            assert cfg is not None
            backend_options = metadata["backends"]
            for backend in backend_options:
                flattened = {f"{backend}.{key}": value for key, value in backend_options[backend].items()}
                cfg.update(flattened)

        # Detect model support code (Allow overwrite in metadata YAML)
        support_path = model_dir / "support"
        if support_path.is_dir():
            assert cfg is not None
            # TODO: onlu overwrite if unset?
            cfg.update({"mlif.model_support_dir": support_path})

        return data_artifact

    def generate_models(self, model):
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

        self.artifacts = artifacts  # If we want to use the same instance of this Frontend in parallel, we need to get rid of self.artifacts...

    def export_models(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_models() first"

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
        path = model.paths[0]
        ext = self.input_formats[0].extension
        with open(path, "rb") as handle:  # TODO: is an onnx model raw data or text?
            raw = handle.read()
            artifacts.append(Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW))
        return artifacts


# TODO: move to frontends.py
# TODO: frontend parsed metadata instead of lookup.py?
# TODO: how to find inout_data?
class TfLiteFrontend(SimpleFrontend):

    FEATURES = Frontend.FEATURES + ["visualize"]

    DEFAULTS = {**Frontend.DEFAULTS, "visualize_graph": False}

    REQUIRED = Frontend.REQUIRED + []

    def __init__(self, features=None, config=None):
        super().__init__(
            "tflite",
            ModelFormats.TFLITE,
            features=features,
            config=config,
        )

    # TODO: ModelFormats.OTHER as placeholder for visualization artifacts


class PackedFrontend(Frontend):  # Inherit from TFLiteFrontend? -> how to do constructor?

    FEATURES = Frontend.FEATURES + ["packing", "packed"]

    DEFAULTS = {
        **Frontend.DEFAULTS,
        "ignore_existing": True,
        "fake_pack": False,  # Pretend that every compatible tensor is packable (best case scenerio, TODO: rename to force_pack?)
        "use_packed": True,
        "check": False,  # Unimplemented
    }

    REQUIRED = ["packer.exe"]  # TODO move to feature?

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
        return bool(self.config["ignore_existing"])

    @property
    def use_packed(self):
        return bool(self.config["fake_pack"])

    @property
    def use_packed(self):
        return bool(self.config["use_packed"])

    @property
    def check(self):
        return bool(self.config["check"])

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
                    tflite_data_in = data_in
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

    FEATURES = Frontend.FEATURES + ["visualize"]

    DEFAULTS = {
        **Frontend.DEFAULTS,
    }

    REQUIRED = Frontend.REQUIRED + []

    def __init__(self, features=None, config=None):
        super().__init__(
            name="onnx",
            input_formats=[ModelFormats.ONNX],
            output_formats=[ModelFormats.ONNX],
            features=features,
            config=config,
        )
