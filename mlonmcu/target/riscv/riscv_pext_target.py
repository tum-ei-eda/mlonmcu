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
"""MLonMCU Spike Target definitions"""


from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from .riscv import RISCVTarget
from .util import update_extensions

logger = get_logger()


class RVPTarget(RISCVTarget):
    """TODO"""

    FEATURES = RISCVTarget.FEATURES | {"pext"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "enable_pext": False,
        "pext_spec": 0.92,
    }
    REQUIRED = RISCVTarget.REQUIRED

    def __init__(
        self,
        name,
        features=None,
        config=None,
    ):
        super().__init__(name, features=features, config=config)
        self.supported_pext_spec_min = 0.92
        self.supported_pext_spec_max = 0.92

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value)

    @property
    def pext_spec(self):
        return float(self.config["pext_spec"])

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(
            exts,
            pext=self.enable_pext,
            pext_spec=self.pext_spec,
        )

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.enable_pext:
            major, minor = str(self.pext_spec).split(".")[:2]
            ret["RISCV_RVP_MAJOR"] = major
            ret["RISCV_RVP_MINOR"] = minor
        return ret
