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
"""TVM Platform"""

from .tvm_base_platform import TvmBasePlatform
from .tvm_target_platform import TvmTargetPlatform
from .tvm_build_platform import TvmBuildPlatform
from .tvm_tune_platform import TvmTunePlatform


class TvmPlatform(
    TvmBasePlatform,
    TvmBuildPlatform,
    # TvmTargetPlatform,  # implicitly via tune platform
    TvmTunePlatform,
):
    """TVM Platform class."""

    FEATURES = (
        TvmBasePlatform.FEATURES | TvmBuildPlatform.FEATURES | TvmTargetPlatform.FEATURES | TvmTunePlatform.FEATURES
    )  # TODO: validate?

    DEFAULTS = {
        **TvmBasePlatform.DEFAULTS,
        **TvmBuildPlatform.DEFAULTS,
        **TvmTargetPlatform.DEFAULTS,
        **TvmTunePlatform.DEFAULTS,
    }

    REQUIRED = (
        TvmBasePlatform.REQUIRED | TvmBuildPlatform.REQUIRED | TvmTargetPlatform.REQUIRED | TvmTunePlatform.REQUIRED
    )

    def __init__(self, features=None, config=None):
        super(TvmPlatform, self).__init__(
            "tvm",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None
