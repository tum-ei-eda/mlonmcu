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
"""MLonMCU ARA Target definitions"""

import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

# import time
import multiprocessing

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool

# from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from .riscv_vext_target import RVVTarget
from .util import update_extensions

logger = get_logger()


class AraTarget(RVVTarget):
    """Target using a Pulpino-like VP running in the GVSOC simulator"""

    FEATURES = RVVTarget.FEATURES | {"log_instrs", "vext"}

    DEFAULTS = {
        **RVVTarget.DEFAULTS,
        "xlen": 64,
        "nr_lanes": 4,
        "vlen": 4096,  # default value for hardware compilation, will be overwritten by -c vext.vlen
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
        "elen": 64,
        "num_threads": multiprocessing.cpu_count(),
        "limit_cycles": 10000000,
    }

    REQUIRED = RVVTarget.REQUIRED | {
        "ara.src_dir",  # for the bsp package
        "verilator.install_dir",  # for simulation
    }

    OPTIONAL = RVVTarget.OPTIONAL | {
        "ara.verilator_tb",
    }

    def __init__(self, name="ara", features=None, config=None):
        super().__init__(name, features=features, config=config)
        assert self.config["xlen"] == 64, 'ARA target must has xlen equal 64, try "-c ara.xlen=64"'

    @property
    def ara_apps_dir(self):
        return Path(self.config["ara.src_dir"]) / "apps"

    @property
    def ara_hardware_dir(self):
        return Path(self.config["ara.src_dir"]) / "hardware"

    @property
    def verilator_install_dir(self):
        return Path(self.config["verilator.install_dir"])

    @property
    def ara_verilator_tb(self):
        value = self.config.get("ara.verilator_tb", None)
        return Path(value) if value is not None else None

    @property
    def nr_lanes(self):
        value = self.config["nr_lanes"]
        return value

    @property
    def vlen(self):
        value = self.config["vlen"]
        return value

    @property
    def limit_cycles(self):
        value = self.config["limit_cycles"]
        return int(value) if value is not None else None

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value)

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value)

    @property
    def elen(self):
        return int(self.config["elen"])

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(
            exts,
        )

    @property
    def num_threads(self):
        return self.config["num_threads"]

    def prepare_simulator(self, program, *_, cwd=os.getcwd(), **kwargs):
        # populate the ara verilator testbench directory
        self.tb_ara_verilator_build_dir = TemporaryDirectory()
        # self.tb_ara_verilator_build_dir = Path("/tmp/tmpd7kz4i6z")
        # env = os.environ.copy()
        # env["ROOT_DIR"] = str(self.ara_hardware_dir)
        # env["veril_library"] = self.tb_ara_verilator_build_dir.name
        # env["veril_path"] = str(self.verilator_install_dir / "bin")
        # env["nr_lanes"] = str(self.nr_lanes)
        # env["vlen"] = str(self.vlen)
        # env["bender_defs"] = f"--define NR_LANES={self.nr_lanes} --define VLEN={self.vlen} --define RVV_ARIANE=1"
        args = []
        args.append(f"ROOT_DIR={self.ara_hardware_dir}")
        args.append(f"veril_library={self.tb_ara_verilator_build_dir.name}")
        args.append(f"veril_path={self.verilator_install_dir}/bin")
        args.append(f"nr_lanes={self.nr_lanes}")
        args.append(f"vlen={self.vlen}")
        compile_verilator_tb_ret = execute(
            "make",
            "verilate",
            f"JOBS={self.num_threads}",
            # env=env,
            cwd=self.ara_hardware_dir,
            *args,
            **kwargs,
        )
        return compile_verilator_tb_ret

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute an executable with given arguments"""
        # run simulation
        # to add trace: https://github.com/pulp-platform/ara/blob/main/hardware/Makefile#L201
        ara_verilator_args = []
        limit_cycles = self.limit_cycles
        if limit_cycles is not None:
            ara_verilator_args.extend(["-c", str(limit_cycles)])
        ara_verilator_args.extend(["-l", f"ram,{program}"])
        assert len(self.extra_args) == 0

        if self.ara_verilator_tb:
            exe = self.ara_verilator_tb
        else:
            assert (
                self.tb_ara_verilator_build_dir is not None
            ), "A folder containing Vara_tb_verilator should be generated by the function prepare_simulator"
            exe = Path(self.tb_ara_verilator_build_dir.name) / "Vara_tb_verilator"
        simulation_ret = execute(
            str(exe),
            *ara_verilator_args,
            # env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        if not self.ara_verilator_tb:
            self.tb_ara_verilator_build_dir.cleanup()
        return simulation_ret

    def parse_exit(self, out):
        exit_code = super().parse_exit(out)
        if exit_code is None:
            if "Simulation timeout of" in out:
                exit_code = -1
        return exit_code

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        """
        Expected output looks like this:

        Simulation statistics
        =====================
        Executed cycles:  5e437
        Wallclock time:   150.173 s
        Simulation speed: 2571.05 cycles/s (2.57105 kHz)
        """
        sim_insns = re.search(r"Executed cycles:\s+([0-9a-f]+)", out)
        if sim_insns:
            sim_insns = int(sim_insns.group(1), 16)
            metrics.add("Simulated Instructions", sim_insns, True)
        wall = re.search(r"Wallclock time:\s+(\d*.?\d*)\ss", out)
        if wall:
            wall = float(wall.group(1))
            metrics.add("Wallclock time", wall, True)
        speed = re.search(r"Simulation speed: (\d*\.?\d*) cycles\/s", out)
        if speed:
            speed = int(float(speed.group(1)))
            metrics.add("MIPS", speed / 1e6, True)

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        if not self.ara_verilator_tb:
            if self.print_outputs:
                self.prepare_simulator(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
            else:
                self.prepare_simulator(
                    elf,
                    *args,
                    cwd=directory,
                    live=False,
                    print_func=lambda *args, **kwargs: None,
                    handle_exit=handle_exit,
                )

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            # TODO: before or after?
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        # simulation_start = time.time()
        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        # simulation_end = time.time()
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)
        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        ret = super().get_platform_defs(platform)
        # ret["RISCV_ARCH"] = "rv32imcxpulpv3"
        ret["XLEN"] = self.xlen
        ret["RISCV_ABI"] = self.abi
        ret["ARA_APPS_DIR"] = self.ara_apps_dir
        ret["MLONMCU_ARA_NR_LANES"] = self.nr_lanes
        ret["MLONMCU_ARA_VLEN"] = self.vlen
        ret["CMAKE_VERBOSE_MAKEFILE"] = "BOOL=OFF"
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        return ret


if __name__ == "__main__":
    cli(target=AraTarget)
