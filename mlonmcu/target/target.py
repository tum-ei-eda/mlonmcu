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
"""MLonMCU Target definitions"""

import os
import tempfile
import time
from pathlib import Path
from typing import List


from mlonmcu.config import filter_config
from mlonmcu.feature.feature import Feature
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.config import str2bool


# TODO: class TargetFactory:
from .common import execute
from .metrics import Metrics


class Target:
    """Base target class

    Attributes
    ----------
    name : str
        Default name of the target
    features : list
        List of target features which should be enabled
    config : dict
        User config defined via key-value pairs
    inspect_program : str
        Program which can be used to inspect executables (i.e. readelf)
    inspect_program_args : list
        List of additional arguments to the inspect_program
    env : os._Environ
        Optinal map of environment variables
    """

    FEATURES = []
    DEFAULTS = {
        "print_outputs": False,
    }

    REQUIRED = []

    def __init__(
        self,
        name: str,
        features: List[Feature] = None,
        config: dict = None,
    ):
        self.name = name
        self.config = config if config else {}
        self.callbacks = []
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)
        self.inspect_program = "readelf"
        self.inspect_program_args = ["--all"]
        self.env = os.environ
        self.artifacts = []

    @property
    def print_outputs(self):
        return str2bool(self.config["print_outputs"])

    def __repr__(self):
        return f"Target({self.name})"

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.TARGET)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            feature.add_target_config(self.name, self.config)
            feature.add_target_callback(self.name, self.callbacks)
        return features

    def exec(self, program: Path, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        raise NotImplementedError

    def inspect(self, program: Path, *args, **kwargs):
        """Use target to inspect a executable"""
        return execute(self.inspect_program, program, *self.inspect_program_args, *args, **kwargs)

    def get_metrics(self, elf, directory, handle_exit=None, num=None):
        assert num is None
        # This should not be accurate, just a fallback which should be overwritten
        start_time = time.time()
        if self.print_outputs:
            out = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out = self.exec(
                elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        # size instead of readelf?
        metrics = Metrics()
        metrics.add("Runtime [s]", diff)

        return metrics, out, []

    def generate_metrics(self, elf, num=None):
        artifacts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, out, artifacts_ = self.get_metrics(elf, temp_dir, num=num)
            artifacts.extend(artifacts_)
            for callback in self.callbacks:  # TODO: give priorities to determine order?
                callback(out, metrics, artifacts)
            content = metrics.to_csv(include_optional=True)  # TODO: store df instead?
            artifact = Artifact("metrics.csv", content=content, fmt=ArtifactFormat.TEXT)
            # Alternative: artifact = Artifact("metrics.csv", data=df/dict, fmt=ArtifactFormat.DATA)
            artifacts.append(artifact)
        stdout_artifact = Artifact(
            f"{self.name}_out.log", content=out, fmt=ArtifactFormat.TEXT
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        self.artifacts = artifacts

    def export_metrics(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_metrics() first"

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

    def get_target_system(self):
        return self.name

    def get_arch(self):
        raise NotImplementedError

    def get_platform_defs(self, platform):
        return {}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))

    def get_backend_config(self, backend):
        return {}
