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
from enum import IntEnum

from mlonmcu.target.target import Target
from mlonmcu.target.host_x86 import HostX86Target
from mlonmcu.target.etiss_pulpino import EtissPulpinoTarget
from mlonmcu.target.corstone300 import Corstone300Target
from mlonmcu.target.spike import SpikeTarget
from mlonmcu.target.ovpsim import OVPSimTarget
from mlonmcu.logging import get_logger

logger = get_logger()

MLIF_TARGET_REGISTRY = {}


def register_mlif_target(target_name, t, override=False):
    global MLIF_TARGET_REGISTRY

    if target_name in MLIF_TARGET_REGISTRY and not override:
        raise RuntimeError(f"MLIF target {target_name} is already registered")
    MLIF_TARGET_REGISTRY[target_name] = t


def get_mlif_targets():
    return MLIF_TARGET_REGISTRY


register_mlif_target("host_x86", HostX86Target)
register_mlif_target("etiss_pulpino", EtissPulpinoTarget)
register_mlif_target("corstone300", Corstone300Target)
register_mlif_target("spike", SpikeTarget)
register_mlif_target("ovpsim", OVPSimTarget)


class MlifExitCode(IntEnum):
    ERROR = 0x10
    INVALID_SIZE = 0x11
    OUTPUT_MISSMATCH = 0x12

    @classmethod
    def values(cls):
        return list(map(int, cls))


def create_mlif_target(name, platform, base=Target):
    class MlifTarget(base):  # This is not ideal as we will have multiple different MlifTarget classes

        FEATURES = base.FEATURES + []

        DEFAULTS = {
            **base.DEFAULTS,
        }
        REQUIRED = base.REQUIRED + []

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.validation_result = None

        def get_metrics(self, elf, directory, handle_exit=None, num=None):
            assert num is None

            # This is wrapper around the original exec function to catch special return codes thrown by the inout data
            # feature (TODO: catch edge cases: no input data available (skipped) and no return code (real hardware))
            if self.platform.validate_outputs and handle_exit is None:

                def _handle_exit(code):
                    if code == 0:
                        self.validation_result = True
                    else:
                        if code in MlifExitCode.values():
                            reason = MlifExitCode(code).name
                            logger.error("A platform error occured during the simulation. Reason: %s", reason)
                            self.validation_result = False
                            if not self.platform.fail_on_error:
                                code = 0
                    return code

                handle_exit = _handle_exit

            metrics, out, artifacts = super().get_metrics(elf, directory, handle_exit=handle_exit)

            if self.platform.validate_outputs:
                metrics.add("Validation", self.validation_result)
            return metrics, out, artifacts

        def get_platform_defs(self, platform):
            ret = super().get_platform_defs(platform)
            target_system = self.get_target_system()
            ret["TARGET_SYSTEM"] = target_system
            return ret

    return MlifTarget
