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
import os
from math import ceil

from mlonmcu.target import Target
from mlonmcu.target.riscv.riscv import RISCVTarget
from mlonmcu.logging import get_logger
from mlonmcu.platform.mlif.mlif_target import MlifExitCode

logger = get_logger()

MLIF_LITEX_PLATFORM_TARGET_REGISTRY = {}


def register_mlif_litex_platform_target(target_name, t, override=False):
    if target_name in MLIF_LITEX_PLATFORM_TARGET_REGISTRY and not override:
        raise RuntimeError(f"MLIF Litex platform target {target_name} is already registered")
    MLIF_LITEX_PLATFORM_TARGET_REGISTRY[target_name] = t


class TemplateMlifLitexPlatformTarget(RISCVTarget):

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "bus_standard": "wishbone",
        "sys_clk_freq": 100e6,
        "integrated_main_ram_size": 0x10000,
        "litex_cpu": None,
        "litex_cpu_variant": "standard",
    }
    REQUIRED = RISCVTarget.REQUIRED

    @property
    def litex_cpu(self):
        value = self.config["litex_cpu"]
        return value

    @property
    def litex_cpu_variant(self):
        value = self.config["litex_cpu_variant"]
        return value

    @property
    def bus_standard(self):
        value = self.config["bus_standard"]
        return value

    @property
    def sys_clk_freq(self):
        value = int(self.config["sys_clk_freq"])
        return value

    @property
    def integrated_main_ram_size(self):
        value = self.config["integrated_main_ram_size"]
        if not isinstance(value, int):
            assert isinstance(value, str)
            value = int(value, 0)
        return value

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)


class VexRiscvTarget(TemplateMlifLitexPlatformTarget):
    DEFAULTS = {
        **TemplateMlifLitexPlatformTarget.DEFAULTS,
        "xlen": 32,
        "atomic": False,
        "compressed": False,
        "fpu": "none",
        "arch": "rv32im",
        "litex_cpu": "vexriscv",
        "litex_cpu_variant": "full",
    }
    REQUIRED = TemplateMlifLitexPlatformTarget.REQUIRED

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)


class CV32E40PTarget(TemplateMlifLitexPlatformTarget):
    DEFAULTS = {
        **TemplateMlifLitexPlatformTarget.DEFAULTS,
        "xlen": 32,
        "atomic": False,
        "compressed": False,
        "fpu": "none",
        "arch": "rv32im",
        "litex_cpu": "cv32e40p",
        "litex_cpu_variant": "standard",
    }
    REQUIRED = TemplateMlifLitexPlatformTarget.REQUIRED

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)


class CVA5Target(TemplateMlifLitexPlatformTarget):
    DEFAULTS = {
        **TemplateMlifLitexPlatformTarget.DEFAULTS,
        "xlen": 32,
        "atomic": False,
        "compressed": False,
        "fpu": "none",
        "arch": "rv32im",
        "litex_cpu": "cva5",
        "litex_cpu_variant": "standard",
    }
    REQUIRED = TemplateMlifLitexPlatformTarget.REQUIRED

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)


def get_mlif_litex_platform_targets():
    return MLIF_LITEX_PLATFORM_TARGET_REGISTRY


register_mlif_litex_platform_target("vexriscv", VexRiscvTarget)
register_mlif_litex_platform_target("cv32e40p", CV32E40PTarget)
register_mlif_litex_platform_target("cva5", CVA5Target)


def create_mlif_litex_platform_target(name, platform, base=Target):
    class MlifLitexPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.validation_result = None

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            # ins_file = None
            # num_inputs = 0
            # in_interface = None
            # out_interface = None
            # batch_size = 1
            # encoding = "utf-8"
            # stdin_data = None
            if self.platform.set_inputs or self.platform.get_outputs:
                raise NotImplementedError
            # outs_file = None
            ret = ""
            artifacts = []
            # num_batches = max(ceil(num_inputs / batch_size), 1)
            if True:
                # ret_, artifacts_ = super().exec(
                #     program, *args, cwd=cwd, **kwargs, stdin_data=stdin_data, encoding=encoding
                # )
                if self.platform.get_outputs:
                    raise NotImplementedError
                ret_, metrics_, artifacts_ = self.platform.run(
                    # program, self, cwd=cwd, ins_file=ins_file, outs_file=outs_file, print_top=print_top
                    program,
                    self,
                    cwd=cwd,
                )
                print("ret_", ret_)
                print("metrics_", metrics_)
                print("artifacts_", artifacts_)
                ret += ret_
                artifacts += artifacts_
            # print("outs_data", outs_data)
            # input("$")
            outs_data = []
            if len(outs_data) > 0:
                raise NotImplementedError
            return ret, artifacts

        def get_metrics(self, elf, directory, handle_exit=None):
            # This is wrapper around the original exec function to catch special return codes thrown by the inout data
            # feature (TODO: catch edge cases: no input data available (skipped) and no return code (real hardware))
            if self.platform.validate_outputs or not self.platform.skip_check:

                def _handle_exit(code, out=None):
                    if handle_exit is not None:
                        code = handle_exit(code, out=out)
                    if code == 0:
                        self.validation_result = True
                    else:
                        if code in MlifExitCode.values():
                            reason = MlifExitCode(code).name
                            logger.error("A platform error occured during the simulation. Reason: %s", reason)
                            if code == MlifExitCode.OUTPUT_MISSMATCH:
                                self.validation_result = False
                                if not self.platform.fail_on_error:
                                    code = 0
                    return code

            else:
                _handle_exit = handle_exit

            metrics, out, artifacts = super().get_metrics(elf, directory, handle_exit=_handle_exit)

            if self.platform.validate_outputs or not self.platform.skip_check:
                metrics.add("Validation", self.validation_result)
            return metrics, out, artifacts

        def get_target_system(self):
            return "litex"

        def get_platform_defs(self, platform):
            ret = super().get_platform_defs(platform)
            target_system = self.get_target_system()
            ret["TARGET_SYSTEM"] = target_system
            return ret

    return MlifLitexPlatformTarget
