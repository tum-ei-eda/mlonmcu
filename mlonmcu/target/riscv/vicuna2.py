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
"""MLonMCU Vicuna Target definitions"""

import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool

# from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from mlonmcu.setup import utils
from .riscv_vext_target import RVVTarget

logger = get_logger()


class Vicuna2Target(RVVTarget):
    """New Vicuna Target with support for (half-precision) scalar/vector FP."""

    FEATURES = RVVTarget.FEATURES | {"log_instrs"}  # TODO: cache feature?, support log_instrs

    DEFAULTS = {
        **RVVTarget.DEFAULTS,
        # vext config
        "vlen": 128,  # default value for hardware compilation, will be overwritten by -c vext.vlen
        "embedded_vext": True,
        "elen": 32,
        # riscv config
        "compressed": False,
        "atomic": False,
        "fpu": None,  # supports: none, half
        # processor config
        "core": "cv32e40x",  # TODO: also support ibex?
        # vproc config
        "mem_size": 4194304,
        "mem_width": 32,
        "vlane_width": 32,
        "vmem_width": 32,
        "mem_latency": 1,
        # cache config
        # "ic_size": 0,  # off
        # "ic_line_width": 128,
        # "dc_size": 0,  # off
        # "dc_line_width": None,  # AUTO (2*VMEM_W)
        # trace config
        # TODO
        # testbench config
        "abort_cycles": 10000000,  # Used to detect freezes
        "extra_cycles": 1,  # Number of remaining cycles after jump to reset vector
        "log_instrs": False,
        "trace": False,
        "trace_full": False,
    }

    REQUIRED = RVVTarget.REQUIRED | {
        "vicuna2.src_dir",
        "verilator.install_dir",  # for simulation
    }

    def __init__(self, name="vicuna2", features=None, config=None):
        super().__init__(name, features=features, config=config)
        assert self.xlen == 32, "Vicuna target must have xlen=32"
        assert not self.enable_vext or self.embedded_vext, "Vicuna target only support embedded vector ext"
        self.prj_dir = None
        self.model_build_dir = None

    @property
    def verilator_install_dir(self):
        return Path(self.config["verilator.install_dir"])

    @property
    def vicuna2_src_dir(self):
        return Path(self.config["vicuna2.src_dir"])

    @property
    def core(self):
        value = self.config["core"]
        assert value == "cv32e40x"
        return value

    @property
    def mem_width(self):
        value = self.config["mem_width"]
        return int(value) if value is not None else value

    @property
    def mem_size(self):
        value = self.config["mem_size"]
        return int(value) if value is not None else value

    @property
    def mem_latency(self):
        value = self.config["mem_latency"]
        return int(value) if value is not None else value

    # @property
    # def ic_size(self):
    #     value = self.config["ic_size"]
    #     return int(value) if value is not None else value

    # @property
    # def ic_line_width(self):
    #     value = self.config["ic_line_width"]
    #     return int(value) if value is not None else value

    # @property
    # def dc_size(self):
    #     value = self.config["dc_size"]
    #     return int(value) if value is not None else value

    # @property
    # def dc_line_width(self):
    #     value = self.config["dc_line_width"]
    #     return int(value) if value is not None else value

    # @property
    # def vproc_config(self):
    #     value = self.config["vproc_config"]
    #     return value

    # @property
    # def vport_policy(self):
    #     value = self.config["vport_policy"]
    #     return value

    @property
    def vmem_width(self):
        value = self.config["vmem_width"]
        return int(value) if value is not None else value

    @property
    def vlane_width(self):
        value = self.config["vlane_width"]
        return int(value) if value is not None else value

    # @property
    # def vproc_pipelines(self):
    #     value = self.config["vproc_pipelines"]
    #     return value

    @property
    def abort_cycles(self):
        value = self.config["abort_cycles"]
        return int(value)

    @property
    def extra_cycles(self):
        value = self.config["extra_cycles"]
        return int(value)

    @property
    def log_instrs(self):
        value = self.config["log_instrs"]
        return str2bool(value)

    @property
    def trace(self):
        value = self.config["trace"]
        return str2bool(value)

    @property
    def trace_full(self):
        value = self.config["trace_full"]
        return str2bool(value)

    def get_model_cmake_args(self):
        ret = []

        def filter_riscv_arch(arch):
            return arch.replace("_zvl128b", "")

        riscv_arch = filter_riscv_arch(self.arch)
        ret.append(f"-DRISCV_ARCH={riscv_arch}")
        ret.append(f"-DVREG_W={self.vlen}")
        if self.vmem_width is not None:
            ret.append(f"-DVMEM_W={self.vmem_width}")
        if self.mem_width is not None:
            ret.append(f"-DMEM_W={self.mem_width}")
        if self.vlane_width is not None:
            ret.append(f"-DVLANE_W={self.vlane_width}")
        # ret.append(f"-DDCACHE_LINE_W={}")
        if self.mem_size is not None:
            # TODO: use in cmake!
            ret.append(f"-DMEM_SZ={self.mem_size}")

        return ret

    def prepare_simulator(self, cwd=os.getcwd(), **kwargs):
        self.prj_dir = TemporaryDirectory()
        self.model_build_dir = Path(self.prj_dir.name) / "build_model"
        self.model_build_dir.mkdir()
        env = os.environ.copy()
        # orig_path = env["PATH"]
        # env["PATH"] = f"{self.verilator_install_dir}/bin:{orig_path}"
        out = utils.cmake(
            self.vicuna2_src_dir / "build_model",
            *self.get_model_cmake_args(),
            env=env,
            cwd=self.model_build_dir,
            **kwargs,
        )
        out += utils.make(env=env, cwd=self.model_build_dir, **kwargs)
        return out

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute an executable with given arguments"""
        vicuna_exe = self.model_build_dir / "verilated_model"
        if len(self.extra_args) > 0:
            assert False, "Vicuna TB does not allow cmdline arguments"

        assert self.prj_dir is not None, "Not prepared?"
        path_file = f"{program}.path"

        instr_trace_file = None
        vcd_trace_file = None
        mem_trace_file = None
        if self.log_instrs:
            instr_trace_file = "instr_trace.csv"
        if self.trace:
            assert self.log_instrs  # Otherwise the number of args will not match!
            mem_trace_file = "mem_trace.csv"
            vcd_trace_file = "rtl_trace.vcd"
        vicuna_args = [
            path_file,
            str(self.mem_width),
            str(self.mem_size),
            str(self.mem_latency),
            str(self.extra_cycles),
            *([instr_trace_file] if self.log_instrs else []),
            *([mem_trace_file, vcd_trace_file] if self.trace else []),
            # trace_vcd,
        ]
        env = os.environ.copy()
        print("vicuna_exe", vicuna_exe)
        print("cwd", cwd)
        input(">")
        out = execute(
            vicuna_exe,
            *vicuna_args,
            env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        input(">")
        # print("prj", self.prj_dir)
        # print("out", out)
        # input("!")
        self.prj_dir.cleanup()
        self.build_dir = None
        return out, []

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0)
        sim_insns = re.search(r"(\d*) cycles", out)
        if sim_insns:
            sim_insns = int(float(sim_insns.group(1)))
            metrics.add("Simulated Instructions", sim_insns, True)

    def parse_exit(self, out):
        exit_code = super().parse_exit(out)
        if exit_code is None:
            if "EXCEPTION: Illegal instruction at" in out:
                exit_code = -4
            elif "WARNING: memory interface inactive for" in out:
                exit_code = -3
            elif "Program finish." not in out:
                exit_code = -2  # did not finish
            # elif "EXCEP" in out:
            #     exit_code = -1  # unhandled exception
        return exit_code

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        artifacts = []

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
            self.prepare_simulator(cwd=directory, live=True)
        else:
            self.prepare_simulator(cwd=directory, live=False, print_func=lambda *args, **kwargs: None)
        simulation_start = time.time()
        if self.print_outputs:
            out_, artifacts_ = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out_, artifacts_ = self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        out += out_
        artifacts += artifacts_
        simulation_end = time.time()
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)
        wall = simulation_end - simulation_start
        # metrics.add("Wallclock time", wall, True)
        if metrics.has("Total Instructions"):
            instructions = metrics.get("Total Instructions")
            if instructions > 0:
                # Warning: Total Instructions != Simulated Instructions
                mips = (instructions / wall) / 1e6
                metrics.add("MIPS", mips, True)
        return metrics, out, artifacts

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["VICUNA2_BSP_DIR"] = self.vicuna2_src_dir / "bsp"
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        return ret


if __name__ == "__main__":
    cli(target=VicunaTarget)
