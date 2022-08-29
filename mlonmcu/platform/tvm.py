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
"""TVM Platform"""

import tempfile
from pathlib import Path

from mlonmcu.setup import utils
from mlonmcu.logging import get_logger
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.target.target import Target
from mlonmcu.config import str2bool
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_bench_tvmc_args,
    get_data_tvmc_args,
    get_rpc_tvmc_args,
    get_target_tvmc_args,
)
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment

from .platform import TargetPlatform, BuildPlatform, TunePlatform
from .tvm_target import create_tvm_platform_target
from .tvm_backend import create_tvm_platform_backend, get_tvm_platform_backends

logger = get_logger()


class TvmPlatform(BuildPlatform, TargetPlatform, TunePlatform):
    """TVM Platform class."""

    FEATURES = TargetPlatform.FEATURES + ["benchmark"]  # TODO: validate?

    DEFAULTS = {
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "fill_mode": "random",
        "ins_file": None,
        "outs_file": None,
        "print_top": False,
        "profile": False,
        "repeat": 1,
        "number": 1,
        "aggregate": "none",  # Allowed: avg, max, min, none, all
        "total_time": False,
        "use_rpc": False,
        "rpc_key": None,
        "rpc_hostname": None,
        "rpc_port": None,
        "tvmc_extra_args": [],
        "tvmc_custom_script": None,
        "experimental_tvmc_tune_tasks": False,
        "experimental_tvmc_tune_visualize": False,
        "visualize_tuning": False,  # Needs upstream patches
        "tune_tasks": None,  # Needs upstream patches
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(
            "tvm",  # Actually: tvmllvm
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    def create_backend(self, name):
        supported = self.get_supported_backends()
        assert name in supported, f"{name} is not a valid TVM platform backend"
        base = supported[name]
        return create_tvm_platform_backend(name, self, base=base)

    @property
    def fill_mode(self):
        return self.config["fill_mode"]

    @property
    def ins_file(self):
        return self.config["ins_file"]

    @property
    def outs_file(self):
        return self.config["outs_file"]

    @property
    def print_top(self):
        return self.config["print_top"]

    @property
    def profile(self):
        value = self.config["profile"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def repeat(self):
        return self.config["repeat"]

    @property
    def number(self):
        return self.config["number"]

    @property
    def aggregate(self):
        value = self.config["aggregate"]
        assert value in ["avg", "all", "max", "min", "none"]
        return value

    @property
    def total_time(self):
        value = self.config["total_time"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def use_rpc(self):
        value = self.config["use_rpc"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def rpc_key(self):
        return self.config["rpc_key"]

    @property
    def rpc_hostname(self):
        return self.config["rpc_hostname"]

    @property
    def rpc_port(self):
        return self.config["rpc_port"]

    @property
    def tvmc_extra_args(self):
        return self.config["tvmc_extra_args"]

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

    @property
    def tvm_pythonpath(self):
        return self.config["tvm.pythonpath"]

    @property
    def tvm_build_dir(self):
        return self.config["tvm.build_dir"]

    @property
    def tvm_configs_dir(self):
        return self.config["tvm.configs_dir"]

    @property
    def experimental_tvmc_tune_tasks(self):
        return str2bool(self.config["experimental_tvmc_tune_tasks"])

    @property
    def experimental_tvmc_tune_visualize(self):
        return str2bool(self.config["experimental_tvmc_tune_visualize"])

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return
        dir_name = self.name
        if path is not None:
            self.project_dir = Path(path)
        elif self.config["project_dir"] is not None:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.debug(
                    "Creating temporary directory because no context was available "
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    def get_supported_backends(self):
        backend_names = get_tvm_platform_backends()
        return backend_names

    def get_supported_targets(self):
        # TODO: get this via tvmc run --help
        target_names = ["cpu", "cuda", "cl", "metal", "vulkan", "rocm", "micro"]

        skip_names = ["micro"]  # Use microtvm platform instead

        return [f"tvm_{name}" for name in target_names if name not in skip_names]

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid TVM device"
        if name in SUPPORTED_TARGETS:
            base = SUPPORTED_TARGETS[name]
        else:
            base = Target
        return create_tvm_platform_target(name, self, base=base)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_tvmc_run_args(self, path, device):
        return [
            path,
            *["--device", device],
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                print_time=True, profile=self.profile, end_to_end=False, repeat=self.repeat, number=self.number
            ),
            *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
        ]

    def invoke_tvmc(self, command, *args):
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=self.print_outputs, print_output=False, env=env)

    def invoke_tvmc_run(self, path, device):
        args = self.get_tvmc_run_args(path, device)
        return self.invoke_tvmc("run", *args)

    def run(self, elf, target, timeout=120):
        # TODO: implement timeout
        # Here, elf is actually a directory
        # TODO: replace workaround with possibility to pass TAR directly
        tar_path = elf
        output = self.invoke_tvmc_run(str(tar_path), target.device)

        return output

    def get_tune_args(self, model, backend, out):

        tuner = backend.config.get("autotuning_tuner", "ga")
        assert tuner in ["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"]
        trials = backend.config.get("autotuning_trials", 10)
        if not isinstance(trials, int):
            trials = int(trials)
        early_stopping = backend.config.get("autotuning_early_stopping", None)
        if early_stopping is None:
            early_stopping = max(trials, 10)  # Let's see if this default works out...
        early_stopping = int(early_stopping)
        # max_parallel = backend.config.get("autotuning_max_parallel", 1)
        # max_parallel = 1
        timeout = backend.config.get("autotuning_timeout", 1000)
        results_file = backend.config.get("autotuning_results_file", None)
        desired_layout = backend.config.get("desired_layout", None)
        ret = [
            *get_target_tvmc_args(
                # "llvm", extra_target=backend.extra_target, target_details=backend.get_target_details()
                "c",
                extra_target=backend.extra_target,
                target_details=backend.get_target_details(),
            ),
            *(["--desired-layout", desired_layout] if desired_layout is not None else []),
            # *get_rpc_tvmc_args(self.use_rpc, self.key, self.hostname, self.port),
            # TODO: missing: pass config, disabled_pass, etc.
            *["--tuner", tuner],
            *(["--early-stopping", str(early_stopping)] if early_stopping > 0 else []),
            # *["--parallel", str(max_parallel)],
            # *["--timeout", str(timeout * max_parallel)],
            *["--timeout", str(timeout)],
            *["--trials", str(trials)],
            # *["--number", str(100)],
            *(["--tuning-records", results_file] if results_file is not None else []),
            *["--output", str(out)],
            # "--target-c-link-params",
            # "1",
            model,
        ]
        if self.visualize_tuning:
            assert (
                self.experimental_tvmc_tune_tasks
            ), f"{self.name}.visualize_tuning requires experimental_autotvm_visualize"
            ret.append("--visualize")
        if self.tune_tasks:
            assert self.experimental_tvmc_tune_tasks, f"{self.name}.tune_tasks requires experimental_tvmc_tune_tasks"
            ret.extend(["--tasks", str(self.tune_tasks)])
        return ret

    def tune_model(self, model_path, backend, target):
        enable = backend.config["autotuning_enable"]
        results_file = backend.config["autotuning_results_file"]
        append = backend.config["autotuning_append"]
        artifacts = []
        if self.print_outputs:
            verbose = True

        content = ""
        if enable:

            if append:
                if results_file is not None:
                    with open(results_file, "r") as handle:
                        content = handle.read()

            with tempfile.TemporaryDirectory() as tmp_dir:
                out_file = Path(tmp_dir) / "tuning_results.log.txt"
                with open(out_file, "w") as handle:
                    handle.write(content)
                    # TODO: newline or not?
                # out = self.invoke_tvmc_tune(out_file, verbose=verbose)
                tune_args = self.get_tune_args(model_path, backend, out_file)
                out = self.invoke_tvmc("tune", tune_args)
                with open(out_file, "r") as handle:
                    content = handle.read()
        else:
            if results_file is None:
                return []
            assert Path(results_file).is_file()
            with open(results_file, "r") as handle:
                content = handle.read()

        artifact = Artifact("tuning_results.log.txt", content=content, fmt=ArtifactFormat.TEXT)
        artifacts.append(artifact)

        # pick best records
        def _pick_best(backend, records, verbose=False):
            content_best = ""
            with tempfile.TemporaryDirectory() as tmp_dir:
                in_file = Path(tmp_dir) / "tuning_results.log.txt"
                with open(in_file, "w") as handle:
                    handle.write(records)
                out_file = Path(tmp_dir) / "best_tuning_results.log.txt"
                args = [
                    "--mode",
                    "pick",
                    "--i",
                    in_file,
                    "--o",
                    out_file,
                ]
                env = prepare_python_environment(backend.tvm_pythonpath, backend.tvm_build_dir, backend.tvm_configs_dir)
                utils.python("-m", "tvm.autotvm.record", *args, live=verbose, env=env)
                with open(out_file, "r") as handle:
                    content_best = handle.read()
            return content_best

        content_best = _pick_best(backend, content, verbose=verbose)
        if len(content_best) > 0:
            artifact_ = Artifact("best_tuning_results.log.txt", content=content_best, fmt=ArtifactFormat.TEXT)
            artifacts.append(artifact_)

        if enable:
            stdout_artifact = Artifact(
                "tvmc_tune_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmaot_out.log?
            artifacts.append(stdout_artifact)

        return artifacts
