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

from mlonmcu.target.riscv.riscv_vext_target import RVVTarget

from mlonmcu.logging import get_logger
from mlonmcu.utils import filter_none

logger = get_logger()

class CanMvK230TvmPlatformTarget(RVVTarget):
    """TODO"""

    FEATURES = RVVTarget.FEATURES | set()

    DEFAULTS = {
        **RVVTarget.DEFAULTS,
        "xlen": 64,
        "vlen": 128,
        "elen": 64,
        "embedded": False,
        "compressed": True,
        "atomic": True,
        "multiply": True,
        "fpu": "double",  # allowed: none, single, double
        "fclk": 1.6e9,
        # "fcpu": 1.6e9,
    }
    REQUIRED = RVVTarget.REQUIRED | set()


    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
