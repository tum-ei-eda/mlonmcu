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

# from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics
from mlonmcu.setup import utils
from .riscv_vext_target import RVVTarget

logger = get_logger()

"""
NOTES
====

# select the core to use as main processor (defaults to Ibex)
CORE     ?= ibex
CORE_DIR := $(SIM_DIR)/../$(CORE)/

# memory initialization files
PROG_PATHS ?= progs.txt

# trace file
TRACE_FILE ?= sim_trace.csv
TRACE_SIGS ?= '*'

"""


class VicunaTarget(RVVTarget):
    """Target using a Pulpino-like VP running in the GVSOC simulator"""

    FEATURES = RVVTarget.FEATURES | {"log_instrs", "vext"}  # TODO: cache feature?

    DEFAULTS = {
        **RVVTarget.DEFAULTS,
        # vext config
        "vlen": 128,  # default value for hardware compilation, will be overwritten by -c vext.vlen
        "embedded_vext": True,
        "elen": 32,
        # riscv config
        "compressed": False,
        "atomic": False,
        "fpu": None,
        # processor config
        "core": "cv32e40x",  # TODO: also support ibex?
        # vector config
        "vproc_config": "compact",  # either compact, dual, triple, legacy, custom
        "vport_policy": None,  # only used for custom config
        "vmem_width": None,
        # "vreg_w": None,  -> determined by vext.vlen!
        "vproc_pipelines": None,
        # memory config
        "mem_width": 32,
        # "mem_size": 262144,
        "mem_size": 262144 * 2,
        "mem_latency": 1,
        # cache config
        "ic_size": 0,  # off
        "ic_line_width": 128,
        "dc_size": 0,  # off
        "dc_line_width": None  # AUTO (2*VMEM_W)
        # trace config
        # TODO
    }

    REQUIRED = RVVTarget.REQUIRED | {
        "vicuna.src_dir",  # for the bsp package
        "verilator.install_dir",  # for simulation
    }

    def __init__(self, name="vicuna", features=None, config=None):
        super().__init__(name, features=features, config=config)
        assert self.xlen == 32, "Vicuna target must have xlen=32"
        assert not self.enable_vext or self.embedded_vext, "Vicuna target only support embedded vector ext"
        self.prj_dir = None
        self.obj_dir = None

    @property
    def verilator_install_dir(self):
        return Path(self.config["verilator.install_dir"])

    @property
    def vicuna_src_dir(self):
        return Path(self.config["vicuna.src_dir"])

    @property
    def core(self):
        value = self.config["core"]
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

    @property
    def ic_size(self):
        value = self.config["ic_size"]
        return int(value) if value is not None else value

    @property
    def ic_line_width(self):
        value = self.config["ic_line_width"]
        return int(value) if value is not None else value

    @property
    def dc_size(self):
        value = self.config["dc_size"]
        return int(value) if value is not None else value

    @property
    def dc_line_width(self):
        value = self.config["dc_line_width"]
        return int(value) if value is not None else value

    @property
    def vproc_config(self):
        value = self.config["vproc_config"]
        return value

    @property
    def vport_policy(self):
        value = self.config["vport_policy"]
        return value

    @property
    def vmem_width(self):
        value = self.config["vmem_width"]
        return int(value) if value is not None else value

    @property
    def vproc_pipelines(self):
        value = self.config["vproc_pipelines"]
        return value

    def get_config_args(self):
        ret = []
        if self.core is not None:
            ret.append(f"CORE={self.core}")
        if self.mem_width is not None:
            ret.append(f"MEM_W={self.mem_width}")
        if self.mem_size is not None:
            ret.append(f"MEM_SZ={self.mem_size}")
        if self.mem_latency is not None:
            ret.append(f"MEM_LATENCY={self.mem_latency}")
        if self.ic_size is not None:
            ret.append(f"ICACHE_SZ={self.ic_size}")
        if self.ic_line_width is not None:
            ret.append(f"ICACHE_LINE_W={self.ic_line_width}")
        if self.dc_size is not None:
            ret.append(f"DCACHE_SZ={self.dc_size}")
        if self.dc_line_width is not None:
            ret.append(f"DCACHE_LINE_W={self.dc_line_width}")
        ret.append(f"VREG_W={self.vlen}")
        if self.vproc_config is not None:
            ret.append(f"VPROC_CONFIG={self.vproc_config}")
        if self.vport_policy is not None:
            ret.append(f"VPORT_POLICY={self.vport_policy}")
        if self.vmem_width is not None:
            ret.append(f"VMEM_W={self.vmem_width}")
        if self.vproc_pipelines is not None:
            ret.append(f"VPROC_PIPELINES={self.vproc_pipelines}")
        return ret

    def prepare_simulator(self, cwd=os.getcwd(), **kwargs):
        # populate the ara verilator testbench directory
        sim_dir = self.vicuna_src_dir / "sim"
        self.prj_dir = TemporaryDirectory()
        self.obj_dir = Path(self.prj_dir.name) / "obj_dir"
        env = os.environ.copy()
        orig_path = env["PATH"]
        # print("self.obj_dir", self.obj_dir)
        # input("000")
        env["PATH"] = f"{self.verilator_install_dir}/bin:{orig_path}"
        out = utils.make("verilator-version-check", env=env, cwd=sim_dir, **kwargs)
        out += utils.make(
            self.obj_dir / "Vvproc_top.mk",
            f"PROJ_DIR={self.prj_dir.name}",
            *self.get_config_args(),
            env=env,
            cwd=sim_dir,
            **kwargs,
        )
        # print("self.obj_dir", self.obj_dir)
        # input("111")
        out += utils.make("-f", self.obj_dir / "Vvproc_top.mk", "Vvproc_top", env=env, cwd=self.obj_dir, **kwargs)
        return out

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute an executable with given arguments"""
        vicuna_exe = self.obj_dir / "Vvproc_top"
        if len(self.extra_args) > 0:
            assert False, "Vicuna TB does not allow cmdline arguments"

        assert self.prj_dir is not None, "Not prepared?"
        path_file = f"{program}.path"

        trace_file = "file"
        trace_vcd = "vcd"
        # trace_fst = "fst"
        vicuna_args = [
            path_file,
            str(self.mem_width),
            str(self.mem_size),
            str(self.mem_latency),
            str(self.vlen * 2),
            trace_file,
            trace_vcd,
        ]
        env = os.environ.copy()
        out = execute(
            vicuna_exe,
            *vicuna_args,
            env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        print("prj", self.prj_dir)
        print("out", out)
        input("!")
        self.prj_dir.cleanup()
        self.obj_dir = None
        return out

    def parse_stdout(self, out):
        cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        if not cpu_cycles:
            logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))

        cpu_instructions = re.search(r"Total Instructions: (.*)", out)
        if not cpu_instructions:
            logger.warning("unexpected script output (instructions)")
            cpu_instructions = None
        else:
            cpu_instructions = int(float(cpu_instructions.group(1)))
        return cycles, cpu_instructions

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        if self.print_outputs:
            self.prepare_simulator(cwd=directory, live=True)
        else:
            self.prepare_simulator(cwd=directory, live=False, print_func=lambda *args, **kwargs: None)
        simulation_start = time.time()
        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        simulation_end = time.time()
        cycles, instructions = self.parse_stdout(out)
        metrics = Metrics()
        metrics.add("Cycles", cycles)
        metrics.add("Instructions", instructions)
        if cycles and instructions:
            metrics.add("CPI", cycles / instructions)
        metrics.add("finished_in_sec", simulation_end - simulation_start, True)
        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["VICUNA_DIR"] = self.vicuna_src_dir
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        return ret


if __name__ == "__main__":
    cli(target=VicunaTarget)
