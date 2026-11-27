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
from mlonmcu.config import cfg, str2bool
from .riscv import RISCVTarget
from .util import update_extensions

logger = get_logger()


class RVBTarget(RISCVTarget):
    """TODO"""

    FEATURES = RISCVTarget.FEATURES | {"bext"}

    enable_bext = cfg(False, cast=str2bool)
    bext_spec = cfg(0.92, cast=float)
    bext_zba = cfg(False, cast=str2bool)
    bext_zbb = cfg(False, cast=str2bool)
    bext_zbc = cfg(False, cast=str2bool)
    bext_zbs = cfg(False, cast=str2bool)

    def __init__(
        self,
        name,
        features=None,
        config=None,
    ):
        super().__init__(name, features=features, config=config)

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(
            exts,
            bext=self.enable_pext,
            bext_spec=self.pext_spec,
            bext_zba=self.bext_zba,
            bext_zbb=self.bext_zbb,
            bext_zbc=self.bext_zbc,
            bext_zbs=self.bext_zbs,
            gcc_major_version=self.gcc_major_version,
        )

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.enable_bext:
            major, minor = str(self.bext_spec).split(".")[:2]
            ret["RISCV_RVB_MAJOR"] = major
            ret["RISCV_RVB_MINOR"] = minor
        return ret
