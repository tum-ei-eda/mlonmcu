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
import re
import tempfile
import time
from pathlib import Path
from typing import List, Tuple


from mlonmcu.config import filter_config
from mlonmcu.utils import filter_none
from mlonmcu.feature.feature import Feature
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.config import str2bool


from mlonmcu.setup.utils import execute
from mlonmcu.target.bench import add_bench_metrics
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

    FEATURES = {"benchmark"}
    DEFAULTS = {
        "print_outputs": False,
        "repeat": None,
    }

    REQUIRED = set()
    OPTIONAL = set()

    def __init__(
        self,
        name: str,
        features: List[Feature] = None,
        config: dict = None,
    ):
        self.name = name
        self.config = config if config else {}
        self.pre_callbacks = []
        self.post_callbacks = []
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        self.inspect_program = "readelf"
        self.inspect_program_args = ["--all"]
        self.env = os.environ
        self.artifacts = []
        self.dir = None

    # def init_directory(self, path=None, context=None):
    #     # return False
    #     assert path is not None
    #         self.dir = Path(path)
    #     self.dir.mkdir(exist_ok=True)

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    @property
    def repeat(self):
        return self.config["repeat"]

    def __repr__(self):
        return f"Target({self.name})"

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.TARGET)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            feature.used = True
            feature.add_target_config(self.name, self.config)
            feature.add_target_callbacks(self.name, self.pre_callbacks, self.post_callbacks)
        return features

    def reconfigure(self):
        pass

    def exec(self, program: Path, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        raise NotImplementedError

    def inspect(self, program: Path, *args, **kwargs):
        """Use target to inspect a executable"""
        return execute(self.inspect_program, program, *self.inspect_program_args, *args, **kwargs)

    def parse_exit(self, out):
        exit_code = None
        exit_match = re.search(r"MLONMCU EXIT: (.*)", out)
        if exit_match:
            exit_code = int(exit_match.group(1))
        return exit_code

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        # This should not be accurate, just a fallback which should be overwritten
        start_time = time.time()

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            # TODO: before or after?
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        if self.print_outputs:
            out, artifacts = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out, artifacts = self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        # size instead of readelf?
        metrics = Metrics()
        metrics.add("End-to-End Runtime [s]", diff)
        exit_code = 0  # TODO: get from handler?
        self.parse_stdout(out, metrics, exit_code=exit_code)

        return metrics, out, artifacts

    def generate(self, elf) -> Tuple[dict, dict]:
        artifacts = []
        metrics = []
        total = 1 + (self.repeat if self.repeat else 0)
        # We only save the stdout and artifacts of the last execution
        # Collect metrics from all runs to aggregate them in a callback with high priority
        artifacts_ = []
        # if self.dir is None:
        #    self.dir = Path(
        with tempfile.TemporaryDirectory() as temp_dir:
            for n in range(total):
                args = []
                for callback in self.pre_callbacks:
                    callback(temp_dir, args, directory=temp_dir)
                if n == total - 1:
                    temp_dir_ = temp_dir
                else:
                    temp_dir_ = Path(temp_dir) / str(n)
                    temp_dir_.mkdir()
                metrics_, out, artifacts_ = self.get_metrics(elf, temp_dir, *args)
                metrics.append(metrics_)
            for callback in self.post_callbacks:
                out = callback(out, metrics, artifacts_, directory=temp_dir)
        artifacts.extend(artifacts_)
        if len(metrics) > 1:
            raise RuntimeError("Collected target metrics for multiple runs. Please aggregate them in a callback!")
        assert len(metrics) == 1
        metrics = metrics[0]
        artifacts_ = {"default": artifacts}
        if not isinstance(metrics, dict):
            metrics = {"default": metrics}
        stdout_artifact = Artifact(
            f"{self.name}_out.log", content=out, fmt=ArtifactFormat.TEXT
        )  # TODO: rename to tvmaot_out.log?
        artifacts_["default"].append(stdout_artifact)
        return artifacts_, metrics

    def generate_artifacts(self, elf):
        start_time = time.time()
        artifacts, metrics = self.generate(elf)
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        for name, metrics_ in metrics.items():
            if name == "default":
                metrics_.add("Run Stage Time [s]", diff, True)
            content = metrics_.to_csv(include_optional=True)  # TODO: store df instead?
            artifact = Artifact("run_metrics.csv", content=content, fmt=ArtifactFormat.TEXT, flags=["metrics"])
            # Alternative: artifact = Artifact("metrics.csv", data=df/dict, fmt=ArtifactFormat.DATA)
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

    def get_target_system(self):
        return self.name

    def get_arch(self):
        raise NotImplementedError

    def get_platform_config(self, platform):
        return {}

    def add_platform_config(self, platform, config):
        config.update(self.get_platform_config(platform))

    def get_platform_defs(self, platform):
        return {}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        return {}

    def add_backend_config(self, backend, config, optimized_layouts=False, optimized_schedules=False):
        new = filter_none(
            self.get_backend_config(
                backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
            )
        )

        # only allow overwriting non-none values
        # to support accepting user-vars
        new = {key: value for key, value in new.items() if config.get(key, None) is None}
        config.update(new)

    def get_hardware_details(self):
        return {
            "num-cores": 1,  # TODO: overwrite for host_x86?
            "cache-line-bytes": 64,  # TODO: disable?
            "vector-unit-bytes": 64,  # TODO: disable?
            # The following are GPU specific
            "max-shared-memory-per-block": 0,
            "max-local-memory-per-block": 0,
            "max-threads-per-block": 0,
            "max-vthread-extent": 0,
            "warp-size": 0,
        }

    @property
    def supports_filesystem(self):
        return False

    @property
    def supports_stdout(self):
        return True

    @property
    def supports_stdin(self):
        return False

    @property
    def supports_argv(self):
        return False

    @property
    def supports_uart(self):
        return False
