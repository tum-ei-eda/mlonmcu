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
"""MicroTVM Platform"""

from mlonmcu.logging import get_logger

from .microtvm_base_platform import MicroTvmBasePlatform
from .microtvm_target_platform import MicroTvmTargetPlatform
from .microtvm_build_platform import MicroTvmBuildPlatform
from .microtvm_tune_platform import MicroTvmTunePlatform
from .microtvm_compile_platform import MicroTvmCompilePlatform

from .microtvm_target import create_microtvm_platform_target, get_microtvm_platform_targets

logger = get_logger()
# TODO: This file is very similar to the TVM platform -> Reuse as much as possible


class MicroTvmPlatform(
    MicroTvmBasePlatform,
    MicroTvmCompilePlatform,
    # MicroTvmTargetPlatform,  # implicitly via tune platform
    MicroTvmBuildPlatform,
    MicroTvmTunePlatform,
):
    """MicroTVM Platform class."""

    FEATURES = (
        MicroTvmBasePlatform.FEATURES
        | MicroTvmCompilePlatform.FEATURES
        | MicroTvmTargetPlatform.FEATURES
        | MicroTvmBuildPlatform.FEATURES
        | MicroTvmTunePlatform.FEATURES
    )  # TODO: validate?

    DEFAULTS = {
        **MicroTvmBasePlatform.DEFAULTS,
        **MicroTvmCompilePlatform.DEFAULTS,
        **MicroTvmTargetPlatform.DEFAULTS,
        **MicroTvmBuildPlatform.DEFAULTS,
        **MicroTvmTunePlatform.DEFAULTS,
    }

    REQUIRED = (
        MicroTvmBasePlatform.REQUIRED
        | MicroTvmCompilePlatform.REQUIRED
        | MicroTvmTargetPlatform.REQUIRED
        | MicroTvmBuildPlatform.REQUIRED
        | MicroTvmTunePlatform.REQUIRED
    )  # TODO: validate?

    def __init__(self, features=None, config=None):
        super(MicroTvmPlatform, self).__init__(
            "microtvm",
            features=features,
            config=config,
        )

    # The following methods are defined here as they would need to be implemented for compile and target platforms
    def get_supported_targets(self):
        return get_microtvm_platform_targets()

    def create_target(self, name):
        supported = self.get_supported_targets()
        assert name in supported, f"{name} is not a valid MicroTVM device"
        base = supported[name]
        return create_microtvm_platform_target(name, self, base=base)
