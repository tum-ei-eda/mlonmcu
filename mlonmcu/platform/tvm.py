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
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.target.target import Target
from mlonmcu.config import str2bool
from mlonmcu.flow.tvm.backend.tvmc_utils import get_bench_tvmc_args, get_data_tvmc_args, get_rpc_tvmc_args
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment

from .platform import TargetPlatform
from .tvm_target import create_tvm_target

logger = get_logger()


class TvmPlatform(TargetPlatform):
    """TVM Platform class."""

    FEATURES = TargetPlatform.FEATURES + []  # TODO: validate?

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
        "use_rpc": False,
        "rpc_key": None,
        "rpc_hostname": None,
        "rpc_port": None,
        "tvmc_extra_args": [],
        "tvmc_custom_script": None,
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
        return str2bool(self.config["profile"])

    @property
    def repeat(self):
        return self.config["repeat"]

    @property
    def use_rpc(self):
        return str2bool(self.config["use_rpc"])

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
        return create_tvm_target(name, self, base=base)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_tvmc_run_args(self, path, device, num=1):
        return [
            path,
            *["--device", device],
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                print_time=True, profile=self.profile, end_to_end=False, repeat=self.repeat, number=num
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

    def invoke_tvmc_run(self, path, device, num=1):
        args = self.get_tvmc_run_args(path, device, num=num)
        return self.invoke_tvmc("run", *args)

    def run(self, elf, target, timeout=120, num=1):
        # TODO: implement timeout
        # Here, elf is actually a directory
        # TODO: replace workaround with possibility to pass TAR directly
        tar_path = elf
        output = self.invoke_tvmc_run(str(tar_path), target.device, num=num)
        return output
