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
import time
import tempfile
import multiprocessing

# from abc import ABC
from abc import abstractmethod
from pathlib import Path
from filelock import FileLock
from typing import Tuple, List

from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features
from mlonmcu.feature.type import FeatureType
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.elf import get_results as get_static_mem_usage
from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from mlonmcu.artifact import Artifact, ArtifactFormat

logger = get_logger()


class Platform:
    """Abstract platform class."""

    FEATURES = set()

    DEFAULTS = {
        "print_outputs": False,
    }

    REQUIRED = set()
    OPTIONAL = set()

    def __init__(self, name, features=None, config=None):
        self.name = name
        self.config = config if config else {}
        self.definitions = {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        self.artifacts = []

    def init_directory(self, path=None, context=None):
        raise NotImplementedError

    @property
    def supports_build(self):
        return False

    @property
    def supports_tune(self):
        return False

    @property
    def supports_compile(self):
        return False

    @property
    def supports_flash(self):
        return False

    @property
    def supports_monitor(self):
        return False

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.PLATFORM)
        for feature in features:
            # assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            if feature.name in self.FEATURES:
                feature.used = True
                feature.add_platform_config(self.name, self.config)
                feature.add_platform_defs(self.name, self.definitions)
        return features

    def get_supported_backends(self):
        return []

    def get_supported_targets(self):
        return []


class BuildPlatform(Platform):
    """Abstract build platform class."""

    @property
    def supports_build(self):
        return True

    def export_artifacts(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_artifacts() first"

        if not isinstance(path, Path):
            path = Path(path)
        assert (
            path.is_dir()
        ), "The supplied path does not exists."  # Make sure it actually exists (we do not create it by default)
        for artifact in self.artifacts:
            artifact.export(path)


class TunePlatform(Platform):
    """Abstract tune platform class."""

    @property
    def supports_tune(self):
        return True

    def export_artifacts(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_artifacts() first"

        if not isinstance(path, Path):
            path = Path(path)
        assert (
            path.is_dir()
        ), "The supplied path does not exists."  # Make sure it actually exists (we do not create it by default)
        for artifact in self.artifacts:
            artifact.export(path)

    @abstractmethod
    def _tune_model(self, model_path, backend, target):
        raise NotImplementedError

    def tune_model(self, model_path, backend, target):
        start_time = time.time()
        artifacts, metrics = self._tune_model(model_path, backend, target)
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        if len(metrics) == 0:
            metrics = {"default": Metrics()}
        for name, metrics_ in metrics.items():
            if name == "default":
                metrics_.add("Tune Stage Time [s]", diff, True)
            content = metrics_.to_csv(include_optional=True)  # TODO: store df instead?
            artifact = Artifact("tune_metrics.csv", content=content, fmt=ArtifactFormat.TEXT, flags=["metrics"])
            if name not in artifacts:
                artifacts[name] = []
            artifacts[name].append(artifact)
        return artifacts


class CompilePlatform(Platform):
    """Abstract compile platform class."""

    FEATURES = Platform.FEATURES | {"debug"}

    DEFAULTS = {
        **Platform.DEFAULTS,
        "debug": False,
        "build_dir": None,
        "num_threads": multiprocessing.cpu_count(),
    }

    @property
    def supports_compile(self):
        return True

    @property
    def debug(self):
        value = self.config["debug"]
        return str2bool(value)

    @property
    def num_threads(self):
        return max(1, int(self.config["num_threads"]))

    def get_metrics(self, elf):
        static_mem = get_static_mem_usage(elf)
        rom_ro, rom_code, rom_misc, ram_data, ram_zdata = (
            static_mem["rom_rodata"],
            static_mem["rom_code"],
            static_mem["rom_misc"],
            static_mem["ram_data"],
            static_mem["ram_zdata"],
        )
        rom_total = rom_ro + rom_code + rom_misc
        ram_total = ram_data + ram_zdata
        metrics = Metrics()
        metrics.add("Total ROM", rom_total)
        metrics.add("Total RAM", ram_total)
        metrics.add("ROM read-only", rom_ro)
        metrics.add("ROM code", rom_code)
        metrics.add("ROM misc", rom_misc)
        metrics.add("RAM data", ram_data)
        metrics.add("RAM zero-init data", ram_zdata)
        return metrics

    @abstractmethod
    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        raise NotImplementedError

    def generate_artifacts(self, src, target, model=None) -> List[Artifact]:
        start_time = time.time()
        artifacts, metrics = self.generate(src, target, model=None)
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        if len(metrics) == 0:
            metrics = {"default": Metrics()}
        for name, metrics_ in metrics.items():
            if name == "default":
                metrics_.add("Compile Stage Time [s]", diff, True)
            content = metrics_.to_csv(include_optional=True)
            artifact = Artifact("compile_metrics.csv", content=content, fmt=ArtifactFormat.TEXT, flags=["metrics"])
            if name not in artifacts:
                artifacts[name] = []
            artifacts[name].append(artifact)
        return artifacts


class TargetPlatform(Platform):
    """Abstract target platform class."""

    def create_target(self, name):
        raise NotImplementedError

    @property
    def supports_flash(self):
        return True

    @property
    def supports_monitor(self):
        return True

    def flash(self, elf, target, timeout=120):
        raise NotImplementedError

    def monitor(self, target, timeout=60):
        raise NotImplementedError

    def run(self, elf, target, timeout=120):
        # Only allow one serial communication at a time
        with FileLock(Path(tempfile.gettempdir()) / "mlonmcu_serial.lock"):
            self.flash(elf, target, timeout=timeout)
            output = self.monitor(target, timeout=timeout)

        return output
