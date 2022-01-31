"""MLonMCU ETISS/Pulpino Target definitions"""

import os
import re
import csv
from pathlib import Path

# from mlonmcu.context import MlonMcuContext
from mlonmcu.logging import get_logger

logger = get_logger()

from .common import cli, execute
from .target import Target
from .metrics import Metrics


# def lookup_riscv_prefix(
#     cfg: dict = None, env: os._Environ = None, context: MlonMcuContext = None
# ) -> Path:
#     """Utility to find the directory where the RISCV GCC compiler toolchain is installed.
#
#     Parameters
#     ----------
#     cfg : dict
#         Optional config provided by the user.
#     env : os._Environ
#         Environment variables
#     context : MlonMcuContext
#         Optional context for looking up dependencies
#
#     Returns
#     -------
#     path : pathlib.Path
#         The path to the toolchain directory (if found).
#     """
#     prefix = None
#
#     if cfg:
#         if "riscv.dir" in cfg:
#             prefix = cfg["riscv.dir"]
#
#     if context:
#         if context.cache:
#             if context.cache["riscv.dir"]:
#                 prefix = context.cache["riscv.dir"]
#
#     if env:
#         if "MLONMCU_HOME" in env:
#             with MlonMcuContext() as ctx:
#                 if ctx.cache:
#                     if ctx.cache["riscv.dir"]:
#                         prefix = ctx.cache["riscv.dir"]
#         elif "RISCV_DIR" in env:
#             prefix = env["RISCV_DIR"]
#
#     if not prefix:
#         prefix = ""
#
#     return prefix
#
#
# def lookup_etiss(
#     cfg: dict = None, env: os._Environ = None, context: MlonMcuContext = None
# ) -> Path:
#     """Utility to find the directory where the ETISS simulator is installed.
#
#     Parameters
#     ----------
#     cfg : dict
#         Optional config provided by the user.
#     env : os._Environ
#         Environment variables
#     context : MlonMcuContext
#         Optional context for looking up dependencies
#
#     Returns
#     -------
#     path : pathlib.Path
#         The path to the ETISS install directory (if found).
#     """
#     etiss = None
#
#     if cfg:
#         if "etiss.dir" in cfg:
#             etiss = cfg["etiss.dir"]
#
#     if context:  # TODO: feature flags?
#         if context.cache:
#             if context.cache["etiss.install_dir"]:
#                 etiss = context.cache["etiss.install_dir"]
#     if env:
#         if "MLONMCU_HOME" in env:
#             with MlonMcuContext() as ctx:
#                 if ctx.cache:
#                     if ctx.cache["etiss.install_dir"]:
#                         etiss = ctx.cache["etiss.install_dir"]
#         if "ETISS_DIR" in env:
#             etiss = env["ETISS_DIR"]
#
#     if not etiss:
#         etiss = ""
#
#     return etiss


class ETISSPulpinoTarget(Target):
    """Target using a Pulpino-like VP running in the ETISS simulator"""

    FEATURES = ["gdbserver", "etissdbg", "trace"]

    DEFAULTS = {
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
        "debug_etiss": False,
        "trace_memory": False,
        "extra_args": "",
        "verbose": False,
        "timeout_sec": 0,  # disabled
        # TODO: how to keep this in sync with setup/tasks.py? (point to ETISSPulpinoTarget.DEFAULTS?)
        "etissvp.rom_start": 0x0,
        "etissvp.rom_size": 0x800000,  # 8 MB
        "etissvp.ram_start": 0x800000,
        "etissvp.ram_size": 0x4000000,  # 64 MB
        "etissvp.cycle_time_ps": 31250,  # 32 MHz
    }
    REQUIRED = ["riscv_gcc.install_dir", "etiss.install_dir"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            "etiss_pulpino", features=features, config=config, context=context
        )
        # self.etiss_dir = lookup_etiss(cfg=config, env=self.env, context=self.context)
        # assert len(self.etiss_dir) > 0
        # self.riscv_prefix = lookup_riscv_prefix(
        #     cfg=config, env=self.env, context=self.context
        # )
        # assert len(self.riscv_prefix) > 0
        self.etiss_script = (
            Path(self.etiss_dir) / "examples" / "bare_etiss_processor" / "run_helper.sh"
        )
        self.metrics_script = (
            Path(self.etiss_dir)
            / "examples"
            / "bare_etiss_processor"
            / "get_metrics.py"
        )
        # TODO: self.cmakeToolchainFile = ?
        # TODO: self.cmakeScript = ?

    @property
    def etiss_dir(self):
        return self.config["etiss.install_dir"]

    @property
    def riscv_prefix(self):
        return self.config["riscv_gcc.install_dir"]

    @property
    def timeout_sec(self):
        # 0 = off
        return int(self.config["timeout_sec"])

    # TODO: add missing properties
    #    "gdbserver_enable": False,
    #    "gdbserver_attach": False,
    #    "gdbserver_port": 2222,
    #    "debug_etiss": False,
    #    "trace_memory": False,
    #    "extra_args": "",
    #    "verbose": True,
    @property
    def rom_start(self):
        return int(self.config["etissvp.rom_start"])

    @property
    def rom_size(self):
        return int(self.config["etissvp.rom_size"])

    @property
    def ram_start(self):
        return int(self.config["etissvp.ram_start"])

    @property
    def ram_size(self):
        return int(self.config["etissvp.ram_size"])

    @property
    def cycle_time_ps(self):
        return int(self.config["etissvp.cycle_time_ps"])

    # TODO: other properties

    def write_ini(self, path):
        with open(path, "w") as f:
            f.write("[IntConfigurations]\n")
            f.write(f"simple_mem_system.memseg_origin_00={hex(self.rom_start)}\n")
            f.write(f"simple_mem_system.memseg_length_00={hex(self.rom_size)}\n")
            f.write(f"simple_mem_system.memseg_origin_01={hex(self.ram_start)}\n")
            f.write(f"simple_mem_system.memseg_length_01={hex(self.ram_size)}\n")
            f.write("\n")
            f.write(f"arch.cpu_cycle_time_ps={self.cycle_time_ps}\n")

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        etiss_script_args = []
        if len(self.config["extra_args"]) > 0:
            etiss_script_args.extend(self.config["extra_args"].split(" "))

        # TODO: this is outdated
        # TODO: validate features (attach xor noattach!)
        if "etissdbg" in self.features:
            etiss_script_args.append("gdb")
        if "attach" in self.features:
            etiss_script_args.append("tgdb")
        if "noattach" in self.features:
            if "attach" not in self.features:
                etiss_script_args.append("tgdb")
            etiss_script_args.append("noattach")
        if "trace" in self.features:
            etiss_script_args.append("trace")
            etiss_script_args.append("nodmi")
        if bool(self.config["verbose"]):
            etiss_script_args.append("v")

        # TODO: working directory?
        etiss_ini = os.path.join(cwd, "custom.ini")
        self.write_ini(etiss_ini)
        etiss_script_args.append("-i" + etiss_ini)

        if self.timeout_sec > 0:
            raise NotImplementedError
        else:
            ret = execute(
                self.etiss_script.resolve(),
                program,
                *etiss_script_args,
                *args,
                **kwargs,
            )
        return ret

    def parse_stdout(self, out):
        exit_match = re.search(r"exit called with code: (.*)", out)
        if exit_match:
            exit_code = exit_match.group(1)
            if int(exit_code) != 0:
                logger.error("Execution failed - " + out)
                raise RuntimeError(f"unexpected exit code: {exit_code}")
        error_match = re.search(r"ETISS: Error: (.*)", out)
        if error_match:
            error_msg = error_match.group(1)
            raise RuntimeError(f"An ETISS Error occured during simulation: {error_msg}")

        cpu_cycles = re.search(r"CPU Cycles \(estimated\): (.*)", out)
        if not cpu_cycles:
            raise RuntimeError("unexpected script output (cycles)")
        cycles = int(float(cpu_cycles.group(1)))
        mips = None  # TODO: parse mips?

        return cycles, mips

    def get_metrics(self, elf, directory, verbose=False):
        if "trace" in self.features:
            trace_file = os.path.join(directory, "dBusAccess.csv")
            if os.path.exists(trace_file):
                os.remove(trace_file)
        else:
            trace_file = None

        metrics_file = os.path.join(directory, "metrics.csv")
        if os.path.exists(metrics_file):
            os.remove(metrics_file)

        if verbose:
            out = self.exec(elf, cwd=directory, live=True)
        else:
            out = self.exec(
                elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None
            )
        total_cycles, mips = self.parse_stdout(out)

        get_metrics_args = [elf]
        etiss_ini = os.path.join(directory, "custom.ini")
        if os.path.exists(etiss_ini):
            get_metrics_args.extend(["--ini", etiss_ini])
        if trace_file:
            get_metrics_args.extend(["--trace", trace_file])
        get_metrics_args.extend(["--out", metrics_file])
        if verbose:
            out2 = execute(self.metrics_script.resolve(), *get_metrics_args, live=True)
        else:
            out2 = execute(
                self.metrics_script.resolve(),
                *get_metrics_args,
                live=False,
                print_func=lambda *args, **kwargs: None,
            )

        metrics = Metrics()
        metrics.add("Total Cycles", total_cycles)
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
                if "trace" in self.features:
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
            if "trace" in self.features:
                metrics.add("RAM stack", ram_stack)
                metrics.add("RAM heap", ram_heap)

        return metrics

    def get_cmake_args(self):
        ret = super().get_cmake_args()
        ret.append(f"-DETISS_DIR={self.etiss_dir}")
        ret.append(f"-DRISCV_ELF_GCC_PREFIX={self.riscv_prefix}")
        return ret


if __name__ == "__main__":
    cli(target=ETISSPulpinoTarget)
