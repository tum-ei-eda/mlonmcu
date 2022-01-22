import tempfile
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Callable
import logging
import copy

from mlonmcu.models.model import Model, ModelFormat

from mlonmcu.logging import get_logger

logger = get_logger()


class Frontend(ABC):

    FEATURES = []

    DEFAULTS = {}

    REQUIRED = []

    def __init__(
        self, name, input_formats=None, output_formats=None, features=None, config=None
    ):
        self.name = name
        self.input_formats = input_formats if input_formats else []
        self.output_formats = output_formats if output_formats else []
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(
            self.config, self.name, self.DEFAULTS, self.REQUIRED
        )

    def supports_formats(ins=None, outs=None):
        """Returs true if the frontend can handle at least one combination of input and output formats."""
        assert (
            ins is not None or outs is not None
        ), "Please provide a list of input formats, outputs formats or both"
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

    def process_features(self):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.TARGET)
        for feature in features:
            assert (  # If this assertion occurs, continue with the next frontend instea dof failing (TODO: create custom exception type)
                feature.name in self.FEATURES
            ), f"Incompatible feature: {feature.name}"
            # Instead we might introduce self.compatible and set it to true at this line
            feature.add_frontend_config(self.name, self.config)
        return features

    @abstractmethod
    def produce_artifacts(self, models, name):
        pass

    def generate_models(path, models):
        artifacts = []

        assert len(models) > 0, f"'{self.name}' frontend expects at least one model"
        max_ins = len(self.input_formats)
        assert (
            len(models) < max_ins
        ), f"'{self.name}' frontend did not expect more than {max_ins} models"
        formats = [model.fmt for model in models]
        assert self.supports_formats(
            formats
        ), f"Invalid model format for '{self.name}' frontend"
        names = [model.name for model in models]
        name = names[0]
        assert all(
            name == name_ for name_ in names
        ), "All input models should share the same name"
        artifacts = self.produce_artifacts(models, name=name)
        if not isinstance(artifacts, list):
            artifacts = [artifacts]
        assert (
            len(artifacts) > 0
        ), f"'{self.name}' frontend should produce at least one model"
        max_outs = len(self.output_formats)
        assert (
            len(artifacts) <= max_outs
        ), f"'{self.name}' frontend should not return more than {max_outs}"

        self.artifacts = artifacts  # If we want to use the same instance of this Frontend in parallel, we need to get rid of self.artifacts...

    def export_models(self, path):
        assert (
            len(self.artifacts) > 0
        ), "No artifacts found, please run generate_models() first"

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

    def produce_artifacts(self, models, name="model"):
        assert len(self.input_formats) == len(self.output_formats) == len(models) == 1
        artifacts = []
        model = models[0]
        ext = self.input_formats[0].extension
        with open(
            model.path, "rb"
        ) as handle:  # TODO: is an onnx model raw data or text?
            data = handle.read()
            artifacts.append(Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW))
        return artifacts


# TODO: move to frontends.py
class TfLiteFrontend(SimpleFrontend):

    FEATURES = Frontend.FEATURES + ["visualize"]

    DEFAULTS = {**Frontend.DEFAULTS, "visualize_graph": False}

    REQUIRED = Frontend.REQUIRED + []

    def __init__(self, features=[], cfg={}):
        super().__init__(
            name="tflite",
            input_formats=[ModelFormat.TFLITE],
            output_formats=[ModelFormat.TFLITE],
            features=features,
            config=config,
        )

    # TODO: ModelFormat.OTHER as placeholder for visualization artifacts


class PackedFrontend(
    Frontend
):  # Inherit from TFLiteFrontend? -> how to do constructor?

    FEATURES = ["packing", "packed"]

    DEFAULTS = {
        **Frontend.DEFAULTS,
        "ignore_existing": True,
        "fake_pack": False,  # Pretend that every compatible tensor is packable (best case scenerio, TODO: rename to force_pack?)
        "use_packed": True,
        "check": False,  # Unimplemented
    }

    REQUIRED = ["packer.exe"]  # TODO move to feature?

    def __init__(self, features=[], cfg={}):
        super().__init__(name="packed", features=features, config=config)
        if self.fake_pack or self.ignore_existing:
            # assert self.use_packed
            self.input_formats = [ModelFormat.TFLITE]
        else:
            self.input_formats = [ModelFormat.PACKED, ModelFormat.TFLITE]

        # if self.use_packed:
        self.output_formats = [
            ModelFormat.PACKED,
            ModelFormat.TFLITE,
        ]  # Always copy over the input model as intermediate artifact for reproducability
        # TODO: add a Frontend.DEFAULT which can be used to turn off this behavoir (keep intermediate?)
        # else:
        #     self.output_formats = [ModelFormat.TFLITE]
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

    def produce_artifacts(self, models, name="model"):
        tflite_data = None
        packed_data = None

        if self.fake_pack or self.ignore_existing:  # -> ins: TFLITE
            # assert self.use_packed
            tflite_model = models[0]

            with open(tflite_model.path, "rb") as handle:
                tflite_data = handle.read()
        else:  # -> ins: TFLITE, PACKED
            for model in models:
                fmt = model.fmt
                data_in = None
                with open(tflite_model.path, "rb") as handle:
                    data_in = handle.read()
                if fmt == ModelFormat.PACKED:
                    packed_data = data_in
                    break
                elif fmt == ModelFormat.TFLITE:
                    tflite_data_in = data_in
                    break
                else:
                    raise RuntimeError(f"Unexpected model format: {fmt}")

        if packed_data is None:
            # Do packing
            with tempfile.TemporaryDirectory() as tmpdirname:
                logger.debug(
                    "Using temporary directory for packing results: %s", tmpdirname
                )
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

    def __init__(self, features=[], cfg={}):
        super().__init__(
            name="onnx",
            input_formats=[ModelFormat.ONNX],
            output_formats=[ModelFormat.ONNX],
            features=features,
            config=config,
        )
