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
from enum import Enum

from mlonmcu.utils import filter_none
from mlonmcu.target.target import Target
from mlonmcu.target.metrics import Metrics

from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.riscv.util import sort_extensions_canonical, join_extensions

from mlonmcu.logging import get_logger

logger = get_logger()


class Esp32C3PerfCount(Enum):
    CYCLE = 1
    INST = 2
    LD_HAZARD = 4
    JMP_HAZARD = 8
    IDLE = 16
    LOAD = 32
    STORE = 64
    JMP_UNCOND = 128
    BRANCH = 256
    BRANCH_TAKEN = 512
    INST_COMP = 1024


ESPIDF_PLATFORM_TARGET_REGISTRY = {}


def register_espidf_platform_target(target_name, t, override=False):
    global ESPIDF_PLATFORM_TARGET_REGISTRY

    if target_name in ESPIDF_PLATFORM_TARGET_REGISTRY and not override:
        raise RuntimeError(f"ESP-IDF platform target {target_name} is already registered")
    ESPIDF_PLATFORM_TARGET_REGISTRY[target_name] = t


def get_espidf_platform_targets():
    return ESPIDF_PLATFORM_TARGET_REGISTRY


class Esp32C3Target(Target):
    DEFAULTS = {
        **Target.DEFAULTS,
        "xlen": 32,
        "extensions": ["i", "m", "c"],
        "fpu": "none",
        "arch": None,
        "abi": None,
        "attr": "",
        "count": Esp32C3PerfCount.CYCLE,
    }

    @property
    def xlen(self):
        return int(self.config["xlen"])

    @property
    def extensions(self):
        exts = self.config.get("extensions", []).copy()
        if not isinstance(exts, list):
            exts = exts.split(",")
        if "g" not in exts:
            required = []
            if self.fpu == "double":
                required.append("d")
                required.append("f")
            if self.fpu == "single":
                required.append("f")
            for ext in required:
                if ext not in exts:
                    exts.append(ext)
        return exts

    @property
    def arch(self):
        temp = self.config["arch"]  # TODO: allow underscores and versions
        if temp:
            return temp
        else:
            exts_str = join_extensions(sort_extensions_canonical(self.extensions, lower=True))
            return f"rv{self.xlen}{exts_str}"

    @property
    def abi(self):
        temp = self.config["abi"]
        if temp:
            return temp
        else:
            if self.xlen == 32:
                temp = "ilp32"
            elif self.xlen == 64:
                temp = "lp64"
            else:
                raise RuntimeError(f"Invalid xlen: {self.xlen}")
            if "d" in self.extensions or "g" in self.extensions:
                temp += "d"
            elif "f" in self.extensions:
                temp += "f"
            return temp

    @property
    def attr(self):
        attrs = str(self.config["attr"]).split(",")
        if len(attrs) == 1 and len(attrs[0]) == 0:
            attrs = []
        for ext in sort_extensions_canonical(self.extensions, lower=True, unpack=True):
            attrs.append(f"+{ext}")
        attrs = list(set(attrs))
        return ",".join(attrs)

    @property
    def fpu(self):
        value = self.config["fpu"]
        if value is None or not value:
            value = "none"
        assert value in ["none", "single", "double"]
        return value

    @property
    def count(self):
        value = int(self.config["count"])
        return value

    @property
    def has_fpu(self):
        return self.fpu != "none"

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["ESP32C3_PERF_COUNT"] = self.count
        return ret

    def get_target_system(self):
        return "esp32c3"

    def get_arch(self):
        return "riscv"

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = {}
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update(
                {
                    "target_device": "riscv_cpu",
                    "target_march": self.arch,
                    "target_model": "esp32c3_devkit",
                    "target_mtriple": "riscv32-esp-elf",
                    "target_mabi": self.abi,
                    "target_mattr": self.attr,
                    "target_mcpu": "esp32c3",
                }
            )
            if optimized_schedules:
                ret.update(
                    {
                        "target_device": "riscv_cpu",
                    }
                )
        return ret

    def add_backend_config(self, backend, config, optimized_layouts=False, optimized_schedules=False):
        new = filter_none(
            self.get_backend_config(
                backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
            )
        )

        # only allow overwriting non-none values
        # to support accepting user-vars
        new = {key: value for key, value in new.items() if config.get(key, None) is None}
        config.update(new)


def create_espidf_platform_target(name, platform, base=Target):
    class EspIdfPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
            "port": None,
            "baud": None,
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform

        @property
        def timeout_sec(self):
            return int(self.config["timeout_sec"])

        @property
        def port(self):
            return self.config["port"]

        @property
        def baud(self):
            return self.config["baud"]

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            """Use target to execute a executable with given arguments"""
            if len(args) > 0:
                raise RuntimeError("Program arguments are not supported for real hardware devices")

            assert self.platform is not None, "ESP32 targets need a platform to execute programs"

            if self.timeout_sec > 0:
                raise NotImplementedError

            # ESP-IDF actually wants a project directory, but we only get the elf now. As a workaround we
            # assume the elf is right in the build directory inside the project directory

            ret = self.platform.run(program, self)
            return ret

        def parse_stdout(self, out):
            cpu_cycles = re.search(r"Total Cycles: (.*)", out)
            if not cpu_cycles:
                logger.warning("unexpected script output (cycles)")
                cycles = None
            else:
                cycles = int(float(cpu_cycles.group(1)))
            cpu_time_us = re.search(r"Total Time: (.*) us", out)
            if not cpu_time_us:
                logger.warning("unexpected script output (time_us)")
                time_us = None
            else:
                time_us = int(float(cpu_time_us.group(1)))
            return cycles, time_us

        def get_metrics(self, elf, directory, handle_exit=None):
            if self.print_outputs:
                out = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
            else:
                out = self.exec(
                    elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
                )
            cycles, time_us = self.parse_stdout(out)

            metrics = Metrics()
            metrics.add("Cycles", cycles)
            time_s = time_us / 1e6 if time_us is not None else time_us
            metrics.add("Runtime [s]", time_s)

            return metrics, out, []

        def get_arch(self):
            return "unkwown"

    return EspIdfPlatformTarget
