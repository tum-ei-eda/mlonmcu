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
import re
import os

from mlonmcu.config import str2bool

# from mlonmcu.target.target import Target
from mlonmcu.target.riscv.riscv import RISCVTarget
from mlonmcu.target.bench import add_bench_metrics

from mlonmcu.logging import get_logger

logger = get_logger()


CFU_PLAYGROUND_PLATFORM_TARGET_REGISTRY = {}


def register_cfu_playground_platform_target(target_name, t, override=False):
    # global CFU_PLAYGROUND_PLATFORM_TARGET_REGISTRY

    if target_name in CFU_PLAYGROUND_PLATFORM_TARGET_REGISTRY and not override:
        raise RuntimeError(f"CFU Pplayground platform target {target_name} is already registered")
    CFU_PLAYGROUND_PLATFORM_TARGET_REGISTRY[target_name] = t


def get_cfu_playground_platform_targets():
    return CFU_PLAYGROUND_PLATFORM_TARGET_REGISTRY


class TemplateCFUPlaygroundPlatformTarget(RISCVTarget):

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)


class FullCFUPlaygroundPlatformTarget(TemplateCFUPlaygroundPlatformTarget):
    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "verbose": False,
        "use_sw_dir": None,
        "use_gateware_dir": None,
        "rtl_sim": False,  # TODO: move to target feature?
        "fpga_sim": False,  # TODO: move to target feature?
        "fpga_tty": None,
        "fpga_target": "digilent_arty",
        "fpga_variant": "a7-100",
        "sys_clk_freq": None,
        "baud": 1843200,  # TODO: fix?
        "cpu_variant": None,
        # Variants:
        # - full+cfu
        # - full+cfu+debug
        # - perf+cfu
        # - perf+cfu+debug
        # - slim+cfu
        # - slim+cfu+debug
        # - slimperf+cfu
        # - slimperf+cfu+debug
        # - minimal+cfu
        # - fpu
        # - fpu+debug
        # - ...
        # Build a custom variant using for example:
        # `-c cfu_full_rtl.cpu_variant="generate+iCacheSize:2048+csrPluginConfig:all+cfu"`
        # Available attributes:
        # csrPluginConfig=mcycle/small/all/linux/linux-minimal
        # bypass=true/false
        # cfu=true/false
        # dCacheSize=4096
        # hardwareDiv=false/true
        # iCacheSize=4096
        # mulDiv=true/false
        # prediction=none/static/dynamic/dynamic_target
        # safe=true/false
        # singleCycleShift=true/false
        # singleCycleMulDiv=true/false
        # debug=true/false
        # UNTESTES:
        # pmpRegions=0?
        # pmpGranularity=256?
        # hardwareBreakpointCount=0
        # atomics=false/true
        # compressedGen=false/true
        # relaxedPcCalculation=false/true
        # externalInterruptArray=true/false
        # resetVector=null?
        # machineTrapVector=null?
        "fpu": "none",  # TODO: use
        "compressed": False,  # TODO: use
        "atomic": False,  # TODO: use
        "integrated_main_ram_size": 0,
        # "integrated_main_ram_size":  256 * 1024,
    }
    REQUIRED = RISCVTarget.REQUIRED | {"tvm.build_dir"}

    @property
    def cpu_variant(self):
        value = self.config["cpu_variant"]
        return value

    @property
    def use_sw_dir(self):
        value = self.config["use_sw_dir"]
        return value

    @property
    def use_gateware_dir(self):
        value = self.config["use_gateware_dir"]
        return value

    @property
    def rtl_sim(self):
        value = self.config["rtl_sim"]
        return str2bool(value)

    @property
    def fpga_sim(self):
        value = self.config["fpga_sim"]
        return str2bool(value)

    @property
    def fpga_tty(self):
        value = self.config["fpga_tty"]
        if value is None:
            raise ValueError("fpga_tty can not be undefined")
        return value

    @property
    def fpga_target(self):
        value = self.config["fpga_target"]
        if value is None:
            raise ValueError("fpga_target can not be undefined")
        return value

    @property
    def fpga_variant(self):
        value = self.config["fpga_variant"]
        if value is None:
            raise ValueError("fpga_variant can not be undefined")
        return value

    @property
    def use_renode(self):
        return not self.rtl_sim and not self.fpga_sim

    @property
    def sys_clk_freq(self):
        value = self.config["sys_clk_freq"]
        if value is None:
            return None
        return int(float(value))

    @property
    def baud(self):
        value = self.config["baud"]
        if value is None:
            return None
        return int(value)

    @property
    def integrated_main_ram_size(self):
        value = self.config["integrated_main_ram_size"]
        if value is None:
            return None
        return int(value)

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)


class FullRTLCFUPlaygroundPlatformTarget(FullCFUPlaygroundPlatformTarget):
    DEFAULTS = {
        **FullCFUPlaygroundPlatformTarget.DEFAULTS,
        "rtl_sim": True,
    }


class FullFPGACFUPlaygroundPlatformTarget(FullCFUPlaygroundPlatformTarget):
    DEFAULTS = {
        **FullCFUPlaygroundPlatformTarget.DEFAULTS,
        "fpga_sim": True,
    }


register_cfu_playground_platform_target("cfu_full", FullCFUPlaygroundPlatformTarget)
register_cfu_playground_platform_target("cfu_full_rtl", FullRTLCFUPlaygroundPlatformTarget)
register_cfu_playground_platform_target("cfu_full_fpga", FullFPGACFUPlaygroundPlatformTarget)


# class DefaultCFUPlaygroundTarget(Target):
#     DEFAULTS = {
#         **Target.DEFAULTS,
#         "xlen": 32,
#         "extensions": ["i", "m", "c"],
#         "fpu": "none",
#         "arch": None,
#         "abi": None,
#         "attr": "",
#     }
#
#     @property
#     def xlen(self):
#         return int(self.config["xlen"])
#
#     @property
#     def extensions(self):
#         exts = self.config.get("extensions", []).copy()
#         if not isinstance(exts, list):
#             exts = exts.split(",")
#         if "g" not in exts:
#             required = []
#             if self.fpu == "double":
#                 required.append("d")
#                 required.append("f")
#             if self.fpu == "single":
#                 required.append("f")
#             for ext in required:
#                 if ext not in exts:
#                     exts.append(ext)
#         return exts
#
#     @property
#     def arch(self):
#         temp = self.config["arch"]  # TODO: allow underscores and versions
#         if temp:
#             return temp
#         else:
#             exts_str = join_extensions(sort_extensions_canonical(self.extensions, lower=True))
#             return f"rv{self.xlen}{exts_str}"
#
#     @property
#     def abi(self):
#         temp = self.config["abi"]
#         if temp:
#             return temp
#         else:
#             if self.xlen == 32:
#                 temp = "ilp32"
#             elif self.xlen == 64:
#                 temp = "lp64"
#             else:
#                 raise RuntimeError(f"Invalid xlen: {self.xlen}")
#             if "d" in self.extensions or "g" in self.extensions:
#                 temp += "d"
#             elif "f" in self.extensions:
#                 temp += "f"
#             return temp
#
#     @property
#     def attr(self):
#         attrs = str(self.config["attr"]).split(",")
#         if len(attrs) == 1 and len(attrs[0]) == 0:
#             attrs = []
#         for ext in sort_extensions_canonical(self.extensions, lower=True, unpack=True):
#             attrs.append(f"+{ext}")
#         attrs = list(set(attrs))
#         return ",".join(attrs)
#
#     @property
#     def fpu(self):
#         value = self.config["fpu"]
#         if value is None or not value:
#             value = "none"
#         assert value in ["none", "single", "double"]
#         return value
#
#     @property
#     def count(self):
#         value = int(self.config["count"])
#         return value
#
#     @property
#     def has_fpu(self):
#         return self.fpu != "none"
#
#     def get_platform_defs(self, platform):
#         ret = super().get_platform_defs(platform)
#         ret["ESP32C3_PERF_COUNT"] = self.count
#         return ret
#
#     def get_target_system(self):
#         return "esp32c3"
#
#     def get_arch(self):
#         return "riscv"
#
#     def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
#         ret = {}
#         if backend in SUPPORTED_TVM_BACKENDS:
#             ret.update(
#                 {
#                     "target_device": "riscv_cpu",
#                     "target_march": self.arch,
#                     "target_model": "esp32c3_devkit",
#                     "target_mtriple": "riscv32-esp-elf",
#                     "target_mabi": self.abi,
#                     "target_mattr": self.attr,
#                     "target_mcpu": "esp32c3",
#                 }
#             )
#             if optimized_schedules:
#                 ret.update(
#                     {
#                         "target_device": "riscv_cpu",
#                     }
#                 )
#         return ret
#
#     def add_backend_config(self, backend, config, optimized_layouts=False, optimized_schedules=False):
#         new = filter_none(
#             self.get_backend_config(
#                 backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
#             )
#         )
#
#         # only allow overwriting non-none values
#         # to support accepting user-vars
#         new = {key: value for key, value in new.items() if config.get(key, None) is None}
#         config.update(new)


def create_cfu_playground_platform_target(name, platform, base=RISCVTarget):
    class CFUPlaygroundPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
            # "port": None,
            # "baud": None,
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform

        @property
        def timeout_sec(self):
            return int(self.config["timeout_sec"])

        # @property
        # def port(self):
        #     return self.config["port"]

        # @property
        # def baud(self):
        #     return self.config["baud"]

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            """Use target to execute a executable with given arguments"""
            if len(args) > 0:
                raise RuntimeError("Program arguments are not supported for real hardware devices")

            assert self.platform is not None, "CFU Playground targets need a platform to execute programs"

            if self.timeout_sec > 0:
                raise NotImplementedError

            # CFU Playground actually wants a project directory, but we only get the elf now. As a workaround we
            # assume the elf is right in the build directory inside the project directory

            out, artifacts, metrics = self.platform.run(program, self)
            return out, artifacts, metrics

        def parse_exit(self, out):
            exit_code = None
            exit_match = re.search(r"MLONMCU EXIT: (.*)", out)
            if exit_match:
                exit_code = int(exit_match.group(1))
            return exit_code

        # def parse_stdout(self, out):
        def parse_stdout(self, out, metrics, exit_code=0):
            add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)

        def get_metrics(self, elf, directory, handle_exit=None):
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
                out, artifacts, metrics = self.exec(elf, cwd=directory, live=True, handle_exit=_handle_exit)
            else:
                out, artifacts, metrics = self.exec(
                    elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
                )
            # metrics = Metrics()
            exit_code = 0  # TODO: get from handler?
            self.parse_stdout(out, metrics, exit_code=exit_code)
            # cycles, time_us = self.parse_stdout(out)
            mips = None
            if metrics.has("Total Instructions") and metrics.has("Simulation Time [s]"):
                sim_insns = metrics.get("Total Instructions")
                sim_time = metrics.get("Simulation Time [s]")
                if sim_time > 0:
                    mips = (sim_insns / sim_time) / 1e6
                    metrics.add("MIPS", mips, True)
            # Add time based on clock freq
            if self.sys_clk_freq:
                for mode in ["Setup", "Run", "Total"]:
                    if metrics.has(f"{mode} Runtime [s]") or metrics.has(f"{mode} Runtime [us]"):
                        continue  # Already populated
                    cycles = metrics.get(f"{mode} Cycles", None)
                    if cycles is None:
                        continue  # Missing cycles
                    secs = cycles / self.sys_clk_freq
                    metrics.add(f"{mode} Runtime [s]", secs, True)

            # metrics = Metrics()
            # metrics.add("Cycles", cycles)
            # time_s = time_us / 1e6 if time_us is not None else time_us
            # metrics.add("Runtime [s]", time_s)

            return metrics, out, artifacts

        def get_arch(self):
            return "unkwown"

    return CFUPlaygroundPlatformTarget
