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
"""TVM Target Platform"""
import os
from mlonmcu.config import str2bool
from .tvm_rpc_platform import TvmRpcPlatform
from ..platform import TargetPlatform
from mlonmcu.target import get_targets
from mlonmcu.target.target import Target
from .tvm_target import create_tvm_platform_target
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_bench_tvmc_args,
    get_data_tvmc_args,
    get_rpc_tvmc_args,
)
from mlonmcu.logging import get_logger

logger = get_logger()


class TvmTargetPlatform(TargetPlatform, TvmRpcPlatform):
    """TVM target platform class."""

    FEATURES = (
        TargetPlatform.FEATURES
        | TvmRpcPlatform.FEATURES
        | {
            "benchmark",
            "tvm_profile",
            "set_inputs",
            "get_outputs",
        }
    )

    DEFAULTS = {
        **TargetPlatform.DEFAULTS,
        **TvmRpcPlatform.DEFAULTS,
        "fill_mode": None,  # random, zeros, ones, none
        "ins_file": None,
        "outs_file": None,
        "print_top": False,
        "profile": False,
        "repeat": 1,
        "number": 1,
        "aggregate": "none",  # Allowed: avg, max, min, none, all
        "total_time": False,
        "set_inputs": False,
        "set_inputs_interface": None,
        "get_outputs": False,
        "get_outputs_interface": None,
        "get_outputs_fmt": None,
    }

    REQUIRED = TargetPlatform.REQUIRED | TvmRpcPlatform.REQUIRED

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
        value = self.config["print_top"]
        return int(value) if isinstance(value, str) else None

    @property
    def profile(self):
        value = self.config["profile"]
        return str2bool(value)

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
        return str2bool(value)

    @property
    def set_inputs(self):
        value = self.config["set_inputs"]
        return str2bool(value)

    @property
    def set_inputs_interface(self):
        value = self.config["set_inputs_interface"]
        return value

    @property
    def get_outputs(self):
        value = self.config["get_outputs"]
        return str2bool(value)

    @property
    def get_outputs_interface(self):
        value = self.config["get_outputs_interface"]
        return value

    @property
    def get_outputs_fmt(self):
        value = self.config["get_outputs_fmt"]  # TODO: use
        return value

    @property
    def inputs_artifact(self):
        # THIS IS A HACK (get inputs fom artifacts!)
        lookup_path = self.project_dir.parent / "inputs.npy"
        if lookup_path.is_file():
            return lookup_path
        else:
            logger.warning("Artifact 'inputs.npz' not found!")
            return None

    def flash(self, elf, target, timeout=120):
        raise NotImplementedError

    def monitor(self, target, timeout=60):
        raise NotImplementedError

    def get_supported_targets(self):
        # TODO: get this via tvmc run --help
        target_names = ["cpu", "cuda", "cl", "metal", "vulkan", "rocm", "micro"]

        skip_names = ["micro"]  # Use microtvm platform instead

        return [f"tvm_{name}" for name in target_names if name not in skip_names]

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid TVM device"
        targets = get_targets()
        if name in targets:
            base = targets[name]
        else:
            base = Target
        return create_tvm_platform_target(name, self, base=base)

    def get_tvmc_run_args(self, ins_file=None, outs_file=None, print_top=None):
        return [
            *get_data_tvmc_args(mode=self.fill_mode, ins_file=ins_file, outs_file=outs_file, print_top=print_top),
            *get_bench_tvmc_args(
                print_time=True, profile=self.profile, end_to_end=False, repeat=self.repeat, number=self.number
            ),
            *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
        ]

    def invoke_tvmc_run(self, *args, target=None, **kwargs):
        assert target is not None, "Target required for tvmc run"
        combined_args = []
        combined_args.extend(["--device", target.device])
        return self.invoke_tvmc("run", *args, **kwargs)

    def run(self, elf, target, timeout=120, cwd=os.getcwd(), ins_file=None, outs_file=None, print_top=None):
        artifacts = []
        # TODO: implement timeout
        # Here, elf is actually a directory
        # TODO: replace workaround with possibility to pass TAR directly
        tar_path = str(elf)
        # in_path = self.ins_file
        # out_path = self.outs_file
        # set_inputs = False
        # if set_inputs and in_path is None:
        #     in_path = Path(cwd) / "ins.npz"
        #     # TODO: populate
        # if self.get_outputs and self.get_outputs_interface == "filesystem" and out_path is None:
        #     out_path = Path(cwd) / "outs.npz"
        args = [tar_path] + self.get_tvmc_run_args(ins_file=ins_file, outs_file=outs_file, print_top=print_top)
        output = self.invoke_tvmc_run(*args, target=target, cwd=cwd)

        return output, artifacts
