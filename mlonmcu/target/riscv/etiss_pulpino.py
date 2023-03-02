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
"""MLonMCU ETISS/Pulpino Target definitions"""

import os
import re
import csv
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool, str2list
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics
from .riscv import RISCVTarget
from .util import update_extensions

logger = get_logger()


class EtissPulpinoTarget(RISCVTarget):
    """Target using a Pulpino-like VP running in the ETISS simulator"""

    FEATURES = RISCVTarget.FEATURES + ["gdbserver", "etissdbg", "trace", "log_instrs", "pext", "vext"]

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
        "debug_etiss": False,
        "trace_memory": False,
        # "plugins": ["PrintInstruction"],
        "plugins": [],
        "verbose": False,
        "cpu_arch": None,
        "rom_start": 0x0,
        "rom_size": 0x800000,  # 8 MB
        "ram_start": 0x800000,
        "ram_size": 0x4000000,  # 64 MB
        "cycle_time_ps": 31250,  # 32 MHz
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
        "enable_pext": False,
        "pext_spec": 0.96,
        "vlen": 0,  # vectorization=off
        "elen": 32,
        "jit": None,
        "end_to_end_cycles": False,
    }
    REQUIRED = RISCVTarget.REQUIRED + ["etiss.src_dir", "etiss.install_dir", "etissvp.script"]

    def __init__(self, name="etiss_pulpino", features=None, config=None):
        super().__init__(name, features=features, config=config)
        self.metrics_script = Path(self.etiss_src_dir) / "src" / "bare_etiss_processor" / "get_metrics.py"

    @property
    def etiss_src_dir(self):
        return self.config["etiss.src_dir"]

    @property
    def etiss_dir(self):
        return self.config["etiss.install_dir"]

    @property
    def etiss_script(self):
        return self.config["etissvp.script"]

    @property
    def gdbserver_enable(self):
        value = self.config["gdbserver_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gdbserver_attach(self):
        value = self.config["gdbserver_attach"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gdbserver_port(self):
        return int(self.config["gdbserver_port"])

    @property
    def debug_etiss(self):
        value = self.config["debug_etiss"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def trace_memory(self):
        value = self.config["trace_memory"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def plugins(self):
        value = self.config["plugins"]
        return str2list(value) if isinstance(value, str) else value

    @property
    def verbose(self):
        value = self.config["verbose"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def rom_start(self):
        value = self.config["rom_start"]
        return int(value, 0) if not isinstance(value, int) else value

    @property
    def rom_size(self):
        value = self.config["rom_size"]
        return int(value, 0) if not isinstance(value, int) else value

    @property
    def ram_start(self):
        value = self.config["ram_start"]
        return int(value, 0) if not isinstance(value, int) else value

    @property
    def ram_size(self):
        value = self.config["ram_size"]
        return int(value, 0) if not isinstance(value, int) else value

    @property
    def cycle_time_ps(self):
        return int(self.config["cycle_time_ps"])

    @property
    def cpu_arch(self):
        if self.config.get("cpu_arch", None):
            return self.config["cpu_arch"]
        elif self.enable_pext or self.enable_vext:
            return "RV32IMACFDPV"
        else:
            return "RV32IMACFD"

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def elen(self):
        return int(self.config["elen"])

    @property
    def jit(self):
        return self.config["jit"]

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(
            exts,
            pext=self.enable_pext,
            pext_spec=self.pext_spec,
            vext=self.enable_vext,
            elen=self.elen,
            embedded=self.embedded_vext,
            fpu=self.fpu,
            variant=self.gcc_variant,
        )

    @property
    def attr(self):
        attrs = super().attr.split(",")
        if self.enable_vext and f"+zvl{self.vlen}b" not in attrs:
            attrs.append(f"+zvl{self.vlen}b")
        return ",".join(attrs)

    @property
    def end_to_end_cycles(self):
        value = self.config["end_to_end_cycles"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def pext_spec(self):
        return float(self.config["pext_spec"])

    def write_ini(self, path):
        # TODO: Either create artifact for ini or prefer to use cmdline args.
        with open(path, "w") as f:
            if self.cpu_arch or self.jit:
                f.write("[StringConfigurations]\n")
            if self.cpu_arch:
                f.write(f"arch.cpu={self.cpu_arch}\n")
            if self.jit:
                f.write(f"jit.type={self.jit}JIT\n")
            f.write("[IntConfigurations]\n")
            # f.write("etiss.max_block_size=100\n")
            # f.write("etiss.max_block_size=500\n")
            f.write(f"simple_mem_system.memseg_origin_00={hex(self.rom_start)}\n")
            f.write(f"simple_mem_system.memseg_length_00={hex(self.rom_size)}\n")
            f.write(f"simple_mem_system.memseg_origin_01={hex(self.ram_start)}\n")
            f.write(f"simple_mem_system.memseg_length_01={hex(self.ram_size)}\n")
            f.write("\n")
            f.write(f"arch.cpu_cycle_time_ps={self.cycle_time_ps}\n")
            if self.has_fpu:
                # TODO: do not hardcode cpu_arch
                # TODO: i.e. use cpu_arch_lower
                f.write("arch.rv32imacfdpv.mstatus_fs=1")
            if self.enable_vext:
                f.write("arch.rv32imacfdpv.mstatus_vs=1")
                if self.vlen > 0:
                    f.write(f"arch.rv32imacfdpv.vlen={self.vlen}")
                # if self.elen > 0:
                #     f.write(f"arch.rv32imacfdpv.elen={self.elen}")

            if self.gdbserver_enable:
                f.write("\n[Plugin gdbserver]\n")
                # This could also be accomplished using `--plugin.gdbserver.port` on the cmdline
                f.write(f"plugin.gdbserver.port={self.gdbserver_port}")

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        etiss_script_args = []
        if len(self.extra_args) > 0:
            etiss_script_args.extend(self.extra_args.split(" "))

        # TODO: this is outdated
        # TODO: validate features (attach xor noattach!)
        if self.debug_etiss:
            etiss_script_args.append("gdb")
        if self.gdbserver_enable:
            etiss_script_args.append("tgdb")
            if not self.gdbserver_attach:
                etiss_script_args.append("noattach")
        if self.trace_memory:
            etiss_script_args.append("trace")
            etiss_script_args.append("nodmi")
        if self.verbose:
            etiss_script_args.append("v")
        # Alternative to stdout parsing: etiss_script_args.append("--vp.stats_file_path=stats.json")

        # TODO: working directory?
        etiss_ini = os.path.join(cwd, "custom.ini")
        self.write_ini(etiss_ini)
        etiss_script_args.append("-i" + etiss_ini)
        for plugin in self.plugins:
            etiss_script_args.extend(["-p", plugin])

        if self.timeout_sec > 0:
            raise NotImplementedError
        else:
            ret = execute(
                Path(self.etiss_script).resolve(),
                program,
                *etiss_script_args,
                *args,
                cwd=cwd,
                **kwargs,
            )
        return ret

    def parse_stdout(self, out, handle_exit=None):
        exit_match = re.search(r"exit called with code: (.*)", out)
        if exit_match:
            exit_code = int(exit_match.group(1))
            if handle_exit is not None:
                exit_code = handle_exit(exit_code)
            if exit_code != 0:
                logger.error("Execution failed - " + out)
                raise RuntimeError(f"unexpected exit code: {exit_code}")
        else:
            exit_code = 0
        error_match = re.search(r"ETISS: Error: (.*)", out)
        if error_match:
            error_msg = error_match.group(1)
            raise RuntimeError(f"An ETISS Error occured during simulation: {error_msg}")

        if self.end_to_end_cycles:
            cpu_cycles = re.search(r"CPU Cycles \(estimated\): (.*)", out)
        else:
            cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        if not cpu_cycles:
            if exit_code == 0:
                logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))
        mips_match = re.search(r"MIPS \(estimated\): (.*)", out)
        if not mips_match:
            if exit_code == 0:
                raise logger.warning("unexpected script output (mips)")
            mips = None
        else:
            mips = int(float(mips_match.group(1)))

        return cycles, mips

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        if self.trace_memory:
            trace_file = os.path.join(directory, "dBusAccess.csv")
            if os.path.exists(trace_file):
                os.remove(trace_file)
        else:
            trace_file = None

        metrics_file = os.path.join(directory, "metrics.csv")
        if os.path.exists(metrics_file):
            os.remove(metrics_file)

        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        total_cycles, mips = self.parse_stdout(out, handle_exit=handle_exit)

        get_metrics_args = [elf]
        etiss_ini = os.path.join(directory, "custom.ini")
        if os.path.exists(etiss_ini):
            get_metrics_args.extend(["--ini", etiss_ini])
        if trace_file:
            get_metrics_args.extend(["--trace", trace_file])
        get_metrics_args.extend(["--out", metrics_file])
        if self.print_outputs:
            out += execute(self.metrics_script.resolve(), *get_metrics_args, live=True)
        else:
            out += execute(
                self.metrics_script.resolve(),
                *get_metrics_args,
                live=False,
                cwd=directory,
                print_func=lambda *args, **kwargs: None,
            )

        metrics = Metrics()
        metrics.add("Cycles", total_cycles)
        metrics.add("MIPS", mips, optional=True)

        metrics_file = os.path.join(directory, "metrics.csv")
        with open(metrics_file, "r") as handle:
            metrics_content = handle.read()
            lines = metrics_content.splitlines()
            reader = csv.DictReader(lines)
            data = list(reader)[0]

            def get_rom_sizes(data):
                assert "rom_rodata" in data
                rom_ro = int(data["rom_rodata"])
                assert "rom_code" in data
                rom_code = int(data["rom_code"])
                assert "rom_misc" in data
                rom_misc = int(data["rom_misc"])

                rom_total = rom_ro + rom_code + rom_misc
                return rom_total, rom_ro, rom_code, rom_misc

            def get_ram_sizes(data):
                assert "ram_data" in data
                ram_data = int(data["ram_data"])
                assert "ram_zdata" in data
                ram_zdata = int(data["ram_zdata"])
                ram_total = ram_data + ram_zdata
                if self.trace_memory:
                    assert "ram_stack" in data
                    ram_stack = int(data["ram_stack"])
                    assert "ram_heap" in data
                    ram_heap = int(data["ram_heap"])
                    ram_total += ram_stack + ram_heap
                else:
                    ram_stack = None
                    ram_heap = None
                return ram_total, ram_data, ram_zdata, ram_stack, ram_heap

            rom_total, rom_ro, rom_code, rom_misc = get_rom_sizes(data)
            ram_total, ram_data, ram_zdata, ram_stack, ram_heap = get_ram_sizes(data)
            metrics.add("Total ROM", rom_total)
            metrics.add("Total RAM", ram_total)
            metrics.add("ROM read-only", rom_ro)
            metrics.add("ROM code", rom_code)
            metrics.add("ROM misc", rom_misc)
            metrics.add("RAM data", ram_data)
            metrics.add("RAM zero-init data", ram_zdata)
            if self.trace_memory:
                metrics.add("RAM stack", ram_stack)
                metrics.add("RAM heap", ram_heap)

        artifacts = []
        ini_content = open(etiss_ini, "r").read()
        ini_artifact = Artifact("custom.ini", content=ini_content, fmt=ArtifactFormat.TEXT)
        artifacts.append(ini_artifact)

        return metrics, out, artifacts

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        ret = super().get_platform_defs(platform)
        ret["ETISS_DIR"] = self.etiss_dir
        ret["PULPINO_ROM_START"] = self.rom_start
        ret["PULPINO_ROM_SIZE"] = self.rom_size
        ret["PULPINO_RAM_START"] = self.ram_start
        ret["PULPINO_RAM_SIZE"] = self.ram_size
        if self.enable_pext:
            major, minor = str(self.pext_spec).split(".")[:2]
            ret["RISCV_RVP_MAJOR"] = major
            ret["RISCV_RVP_MINOR"] = minor
        if self.enable_vext:
            major, minor = str(self.vext_spec).split(".")[:2]
            ret["RISCV_RVV_MAJOR"] = major
            ret["RISCV_RVV_MINOR"] = minor
            ret["RISCV_RVV_VLEN"] = self.vlen
        return ret

    def get_backend_config(self, backend):
        ret = super().get_backend_config(backend)
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update({"target_model": "etissvp"})
            if self.enable_pext or self.enable_vext:
                ret.update(
                    {
                        # Warning: passing kernel layouts does not work with upstream TVM
                        # TODO: allow passing map?
                        "desired_layout": "NHWC:HWOI",
                    }
                )
        return ret


if __name__ == "__main__":
    cli(target=EtissPulpinoTarget)
