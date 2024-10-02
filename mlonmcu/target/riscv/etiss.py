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
from mlonmcu.timeout import exec_timeout
from mlonmcu.config import str2bool, str2list, str2dict
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from .riscv import RISCVTarget

logger = get_logger()


class EtissTarget(RISCVTarget):
    """Target using a simple RISC-V VP running in the ETISS simulator"""

    FEATURES = RISCVTarget.FEATURES | {
        "gdbserver",
        "etissdbg",
        "trace",
        "log_instrs",
        "pext",
        "vext",
        "xcorev",
        "vanilla_accelerator",
    }

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
        "rom_start": 0x1000000,
        "rom_size": 0x800000,  # 8 MB
        "ram_start": 0x1000000 + 0x800000,
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
        "allow_error": False,
        "max_block_size": None,
        "enable_xcorevmac": False,
        "enable_xcorevmem": False,
        "enable_xcorevbi": False,
        "enable_xcorevalu": False,
        "enable_xcorevbitmanip": False,
        "enable_xcorevsimd": False,
        "enable_xcorevhwlp": False,
        "extra_int_config": {},
        "extra_bool_config": {},
        "extra_string_config": {},
        "extra_plugin_config": {},
        "use_run_helper": True,
        "exit_on_loop": False,
        "log_pc": False,
        "log_level": None,
        "enable_semihosting": True,
        "output_path_prefix": "",
        "jit_gcc_cleanup": True,
        "jit_verify": False,
        "jit_debug": False,
        "load_integrated_libraries": True,
    }
    REQUIRED = RISCVTarget.REQUIRED | {"etiss.src_dir", "etiss.install_dir", "etissvp.exe", "etissvp.script"}

    def __init__(self, name="etiss", features=None, config=None):
        super().__init__(name, features=features, config=config)
        # TODO: make optional or move to mlonmcu pkg
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
    def etiss_exe(self):
        return self.config["etissvp.exe"]

    @property
    def gdbserver_enable(self):
        value = self.config["gdbserver_enable"]
        return str2bool(value)

    @property
    def gdbserver_attach(self):
        value = self.config["gdbserver_attach"]
        return str2bool(value)

    @property
    def gdbserver_port(self):
        return int(self.config["gdbserver_port"])

    @property
    def debug_etiss(self):
        value = self.config["debug_etiss"]
        return str2bool(value)

    @property
    def trace_memory(self):
        value = self.config["trace_memory"]
        return str2bool(value)

    @property
    def enable_dmi(self):
        return False
        # return not self.trace_memory

    @property
    def plugins(self):
        value = self.config["plugins"]
        return str2list(value)

    def get_plugin_names(self):
        ret = self.plugins
        if self.gdbserver_enable:
            ret.append("gdbserver")
        return list(set(ret))

    @property
    def verbose(self):
        value = self.config["verbose"]
        return str2bool(value)

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
            return f"RV{self.xlen}IMACFDPV"
        else:
            return f"RV{self.xlen}IMACFD"

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value)

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value)

    @property
    def enable_xcorevmac(self):
        value = self.config["enable_xcorevmac"]
        return str2bool(value)

    @property
    def enable_xcorevmem(self):
        value = self.config["enable_xcorevmem"]
        return str2bool(value)

    @property
    def enable_xcorevbi(self):
        value = self.config["enable_xcorevbi"]
        return str2bool(value)

    @property
    def enable_xcorevalu(self):
        value = self.config["enable_xcorevalu"]
        return str2bool(value)

    @property
    def enable_xcorevbitmanip(self):
        value = self.config["enable_xcorevbitmanip"]
        return str2bool(value)

    @property
    def enable_xcorevsimd(self):
        value = self.config["enable_xcorevsimd"]
        return str2bool(value)

    @property
    def enable_xcorevhwlp(self):
        value = self.config["enable_xcorevhwlp"]
        return str2bool(value)

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
    def extra_bool_config(self):
        value = self.config["extra_bool_config"]
        return str2dict(value)

    @property
    def extra_int_config(self):
        value = self.config["extra_int_config"]
        return str2dict(value)

    @property
    def extra_string_config(self):
        value = self.config["extra_string_config"]
        return str2dict(value)

    @property
    def extra_plugin_config(self):
        value = self.config["extra_plugin_config"]
        return str2dict(value)

    @property
    def extensions(self):
        exts = super().extensions
        required = set()
        if "xcorev" not in exts:
            if self.enable_xcorevmac:
                required.add("xcvmac")
            if self.enable_xcorevmem:
                required.add("xcvmem")
            if self.enable_xcorevbi:
                required.add("xcvbi")
            if self.enable_xcorevalu:
                required.add("xcvalu")
            if self.enable_xcorevbitmanip:
                required.add("xcvbitmanip")
            if self.enable_xcorevsimd:
                required.add("xcvsimd")
            if self.enable_xcorevhwlp:
                required.add("xcvhwlp")
        for ext in required:
            if ext not in exts:
                exts.add(ext)
        return exts

    @property
    def attr(self):
        attrs = super().attr.split(",")
        # attrs.append("+unaligned-scalar-mem")
        # attrs = [x for x in attrs if x != "+c"]
        # attrs.append("-c")
        if self.enable_xcorevmac:
            if "xcorevmac" not in attrs:
                attrs.append("+xcvmac")
        if self.enable_xcorevmem:
            if "xcorevmem" not in attrs:
                attrs.append("+xcvmem")
        if self.enable_xcorevbi:
            if "xcorevbi" not in attrs:
                attrs.append("+xcvbi")
        if self.enable_xcorevalu:
            if "xcorevalu" not in attrs:
                attrs.append("+xcvalu")
        if self.enable_xcorevbitmanip:
            if "xcorevbitmanip" not in attrs:
                attrs.append("+xcvbitmanip")
        if self.enable_xcorevsimd:
            if "xcorevsimd" not in attrs:
                attrs.append("+xcvsimd")
        if self.enable_xcorevhwlp:
            if "xcorevhwlp" not in attrs:
                attrs.append("+xcvhwlp")
        if self.enable_vext and f"+zvl{self.vlen}b" not in attrs:
            attrs.append(f"+zvl{self.vlen}b")
        return ",".join(attrs)

    @property
    def allow_error(self):
        value = self.config["allow_error"]
        return str2bool(value)

    @property
    def use_run_helper(self):
        value = self.config["use_run_helper"]
        return str2bool(value)

    @property
    def exit_on_loop(self):
        value = self.config["exit_on_loop"]
        return str2bool(value)

    @property
    def log_pc(self):
        value = self.config["log_pc"]
        return str2bool(value)

    @property
    def log_level(self):
        value = self.config["log_level"]
        if isinstance(value, str):
            value = int(value)
        return value

    @property
    def enable_semihosting(self):
        value = self.config["enable_semihosting"]
        return str2bool(value)

    @property
    def output_path_prefix(self):
        return self.config["output_path_prefix"]

    @property
    def jit_gcc_cleanup(self):
        value = self.config["jit_gcc_cleanup"]
        return str2bool(value)

    @property
    def jit_verify(self):
        value = self.config["jit_verify"]
        return str2bool(value)

    @property
    def jit_debug(self):
        value = self.config["jit_debug"]
        return str2bool(value)

    @property
    def load_integrated_libraries(self):
        value = self.config["load_integrated_libraries"]
        return str2bool(value)

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value)

    @property
    def pext_spec(self):
        return float(self.config["pext_spec"])

    @property
    def max_block_size(self):
        value = self.config["max_block_size"]
        if isinstance(value, str):
            value = int(value)
        return value

    def get_ini_bool_config(self, override=None):
        override = {k: v for k, v in override.items() if isinstance(v, bool)}

        ret = {
            "arch.enable_semihosting": self.enable_semihosting,
            "simple_mem_system.error_on_invalid_access": not self.allow_error,
            "jit.verify": self.jit_verify,
            "jit.debug": self.jit_debug,
            "etiss.load_integrated_libraries": self.load_integrated_libraries,
        }
        if not self.use_run_helper:
            ret["simple_mem_system.print_dbus_access"] = self.trace_memory
            ret["simple_mem_system.print_to_file"] = self.trace_memory
            ret["etiss.exit_on_loop"] = self.exit_on_loop
            ret["etiss.log_pc"] = self.log_pc
            ret["etiss.enable_dmi"] = self.enable_dmi
        if self.jit == "GCC":
            ret["jit.gcc.cleanup"] = self.jit_gcc_cleanup

        ret.update(self.extra_bool_config)
        ret.update(override)
        return ret

    def get_ini_string_config(self, override=None):
        override = {k: v for k, v in override.items() if isinstance(v, (str, Path))}
        ret = {
            "arch.cpu": self.cpu_arch,
            # Mode will be overwritten by elf...
            # "simple_mem_system.memseg_mode_00": "RX",
            # "simple_mem_system.memseg_mode_01": "RWX",
            "etiss.output_path_prefix": self.output_path_prefix,
        }
        if self.jit is not None:
            ret["jit.type"] = f"{self.jit}JIT"
        ret.update(self.extra_string_config)
        ret.update(override)
        return ret

    def get_ini_int_config(self, override=None):
        override = {k: v for k, v in override.items() if isinstance(v, int)}
        ret = {
            "simple_mem_system.memseg_origin_00": self.rom_start,
            "simple_mem_system.memseg_length_00": self.rom_size,
            "simple_mem_system.memseg_origin_01": self.ram_start,
            "simple_mem_system.memseg_length_01": self.ram_size,
            "arch.cpu_cycle_time_ps": self.cycle_time_ps,
        }
        if self.max_block_size:
            ret["etiss.max_block_size"] = self.max_block_size
        if self.has_fpu:
            # TODO: do not hardcode cpu_arch
            # TODO: i.e. use cpu_arch_lower
            ret["arch.rv32imacfdpv.mstatus_fs"] = 1
        if self.enable_vext:
            ret["arch.rv32imacfdpv.mstatus_vs"] = 1
            if self.vlen > 0:
                ret["arch.rv32imacfdpv.vlen"] = self.vlen
            if self.elen > 0:
                ret["arch.rv32imacfdpv.elen"] = self.elen
        log_level = self.log_level
        if log_level is None and not self.use_run_helper:
            log_level = 5 if self.verbose else 4
        if log_level is not None:
            ret["etiss.loglevel"] = log_level
        # TODO
        # ETISS::CPU_quantum_ps=100000
        # ETISS::write_pc_trace_from_time_us=0
        # ETISS::write_pc_trace_until_time_us=3000000
        # ETISS::sim_mode=0
        # vp::simulation_time_us=20000000
        ret.update(self.extra_int_config)
        ret.update(override)
        return ret

    def get_ini_plugin_config(self):
        ret = {}
        if self.gdbserver_enable:
            if not self.use_run_helper:
                ret["gdbserver"] = {
                    "port": self.gdbserver_port,
                }
        ret.update(self.extra_plugin_config)  # TODO: merge nested dict instead of overriding
        return ret

    def write_ini(self, path, override=None):
        # TODO: Either create artifact for ini or prefer to use cmdline args.
        with open(path, "w") as f:
            ini_bool = self.get_ini_bool_config(override=override)
            if len(ini_bool) > 0:
                f.write("[BoolConfigurations]\n")
                for key, value in ini_bool.items():
                    assert isinstance(value, bool)
                    val = "true" if value else "false"
                    f.write(f"{key}={val}\n")
            ini_string = self.get_ini_string_config(override=override)
            if len(ini_string) > 0:
                f.write("[StringConfigurations]\n")
                for key, value in ini_string.items():
                    if isinstance(value, Path):
                        value = str(value)
                    assert isinstance(value, str)
                    f.write(f"{key}={value}\n")
            ini_int = self.get_ini_int_config(override=override)
            if len(ini_int) > 0:
                f.write("[IntConfigurations]\n")
                for key, value in ini_int.items():
                    assert isinstance(value, int)
                    f.write(f"{key}={value}\n")
            ini_plugin = self.get_ini_plugin_config()
            for plugin_name in self.get_plugin_names():
                f.write(f"[Plugin {plugin_name}]\n")
                cfg = ini_plugin.pop(plugin_name, {})
                for key, value in cfg.items():
                    if isinstance(value, bool):
                        val = "true" if value else "false"
                    else:
                        val = value
                    f.write(f"plugin.{plugin_name}.{key}={val}\n")
            # Check for remaining configs
            for plugin_name, cfg in ini_plugin.items():
                if len(cfg) == 0:
                    continue
                logger.warning("Skipping config %s for disabled plugin %s", cfg, plugin_name)

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        etiss_script_args = []
        if len(self.extra_args) > 0:
            if not self.use_run_helper:
                raise NotImplementedError("etiss.extra_args requires etiss.use_run_helper=1")
            etiss_script_args.extend(self.extra_args.split(" "))

        # TODO: this is outdated
        # TODO: validate features (attach xor noattach!)
        if self.debug_etiss:
            if self.use_run_helper:
                etiss_script_args.append("gdb")
            else:
                raise NotImplementedError("etiss.debug_etiss requires etiss.use_run_helper=1")
        if self.gdbserver_enable:
            if self.use_run_helper:
                etiss_script_args.append("tgdb")
                if not self.gdbserver_attach:
                    etiss_script_args.append("noattach")
                etiss_script_args.append("--plugin.gdbserver.port={self.gdbserver_port}")
                if self.gdbserver_attach:
                    raise NotImplementedError("etiss.gdbserver_attach requires etiss.use_run_helper=1")
        if self.trace_memory:
            if self.use_run_helper:
                etiss_script_args.append("trace")
                etiss_script_args.append("nodmi")
        if self.exit_on_loop:
            if self.use_run_helper:
                etiss_script_args.append("noloop")
        if self.log_pc:
            if self.use_run_helper:
                etiss_script_args.append("logpc")
        if not self.enable_dmi:
            if self.use_run_helper:
                etiss_script_args.append("nodmi")
        if self.verbose:
            etiss_script_args.append("v")
        # Alternative to stdout parsing: etiss_script_args.append("--vp.stats_file_path=stats.json")
        if self.use_run_helper:
            for plugin in self.get_plugin_names():
                etiss_script_args.extend(["-p", plugin])

        # TODO: working directory?
        ini_override = {}
        if self.use_run_helper:
            etiss_script_args.insert(0, program)
        else:
            ini_override["vp.elf_file"] = program
        etiss_ini = os.path.join(cwd, "custom.ini")
        self.write_ini(etiss_ini, override=ini_override)
        etiss_script_args.append("-i" + etiss_ini)

        # if self.timeout_sec > 0:
        script = self.etiss_script if self.use_run_helper else self.etiss_exe
        script = Path(script).resolve()
        if False:
            ret = exec_timeout(
                self.timeout_sec,
                execute,
                script,
                *etiss_script_args,
                *args,
                cwd=cwd,
                **kwargs,
            )
        else:
            ret = execute(
                script,
                *etiss_script_args,
                *args,
                cwd=cwd,
                **kwargs,
            )
        return ret, []

    def parse_exit(self, out):
        exit_code = super().parse_exit(out)
        if exit_code is None:
            # legacy
            exit_match = re.search(r"exit called with code: (.*)", out)
            if exit_match:
                exit_code = int(exit_match.group(1))
        return exit_code

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        error_match = re.search(r"ETISS: Error: (.*)", out)
        if error_match:
            error_msg = error_match.group(1)
            if self.allow_error:
                logger.error(f"An ETISS Error occured during simulation: {error_msg}")
            else:
                raise RuntimeError(f"An ETISS Error occured during simulation: {error_msg}")
        sim_insns = re.search(r"CPU Cycles \(estimated\): (.*)", out)
        sim_insns = int(float(sim_insns.group(1)))
        metrics.add("Simulated Instructions", sim_insns, True)
        mips = None  # TODO: parse mips?
        mips_match = re.search(r"MIPS \(estimated\): (.*)", out)
        if mips_match:
            mips_str = mips_match.group(1)
            mips = float(mips_str)
        if mips:
            metrics.add("MIPS", mips, optional=True)

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

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            # TODO: before or after?
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        artifacts = []

        if self.print_outputs:
            out_, artifacts_ = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
            out += out_
            artifacts += artifacts_
        else:
            out_, artifacts_ = self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
            out += out_
            artifacts += artifacts_
        # TODO: get exit code
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)

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

        ini_content = open(etiss_ini, "r").read()
        ini_artifact = Artifact("custom.ini", content=ini_content, fmt=ArtifactFormat.TEXT)
        artifacts.append(ini_artifact)

        return metrics, out, artifacts

    def get_target_system(self):
        if not self.enable_semihosting:
            raise NotImplementedError("etiss.enable_semihosting=0 is not supported anymore")
        return self.name

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        assert self.is_bare, "ETISS Target needs baremetal toolchain (riscv[32|64]-unknown-elf)"
        ret = super().get_platform_defs(platform)
        ret["MEM_ROM_ORIGIN"] = self.rom_start
        ret["MEM_ROM_LENGTH"] = self.rom_size
        ret["MEM_RAM_ORIGIN"] = self.ram_start
        ret["MEM_RAM_LENGTH"] = self.ram_size
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

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        if backend in SUPPORTED_TVM_BACKENDS:
            if optimized_layouts:
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
    cli(target=EtissTarget)
