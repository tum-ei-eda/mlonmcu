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

from mlonmcu.target import get_targets, Target
from mlonmcu.logging import get_logger

logger = get_logger()


def get_mlif_platform_targets():
    return get_targets()


class MlifExitCode(IntEnum):
    ERROR = 0x10
    INVALID_SIZE = 0x11
    OUTPUT_MISSMATCH = 0x12

    @classmethod
    def values(cls):
        return list(map(int, cls))


def create_mlif_platform_target(name, platform, base=Target):
    class MlifPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.validation_result = None

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

        def get_platform_defs(self, platform):
            ret = super().get_platform_defs(platform)
            target_system = self.get_target_system()
            ret["TARGET_SYSTEM"] = target_system
            return ret

    return MlifPlatformTarget
