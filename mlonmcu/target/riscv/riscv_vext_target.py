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
from mlonmcu.utils import is_power_of_two
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from .riscv import RISCVTarget
from .util import update_extensions

logger = get_logger()


class RVVTarget(RISCVTarget):
    """TODO"""

    FEATURES = RISCVTarget.FEATURES | {"vext"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
        "vlen": 128,  # vectorization=off
        "elen": 64,
    }
    REQUIRED = RISCVTarget.REQUIRED

    def __init__(self, name, features=None, config=None):
        super().__init__(name, features=features, config=config)
        self.supported_vext_spec_min = 1.0
        self.supported_vext_spec_max = 1.0

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value)

    @property
    def vlen(self):
        value = int(self.config["vlen"])
        assert value == 0 or is_power_of_two(value), "VLEN needs to be a power of 2."
        assert value == 0 or value >= 32, "VLEN < 32 not allowed"
        if 0 < value < 128:
            assert self.embedded_vext, "VLEN < 128 imples embedded_vext=false"
        return value

    @property
    def elen(self):
        value = int(self.config["elen"])
        assert value in [32, 64]
        if value == 32:
            assert self.embedded_vext, "ELEN=32 imples embedded_vext=true"
        return value

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value)

    @property
    def extensions(self):
        exts = super().extensions
        assert (
            self.supported_vext_spec_min <= self.vext_spec <= self.supported_vext_spec_max
        ), f"V-Extension spec {self.vext_spec} not supported by {self.name} target"
        return update_extensions(
            exts,
            vext=self.enable_vext,
            elen=self.elen,
            embedded_vext=self.embedded_vext,
            vlen=self.vlen,
            fpu=self.fpu,
        )

    @property
    def attrs(self):
        attrs = super().attrs
        if self.enable_vext and f"+zvl{self.vlen}b" not in attrs:
            attrs.append(f"+zvl{self.vlen}b")
        return attrs

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.enable_vext:
            major, minor = str(self.vext_spec).split(".")[:2]
            ret["RISCV_RVV_MAJOR"] = major
            ret["RISCV_RVV_MINOR"] = minor
            ret["RISCV_RVV_VLEN"] = self.vlen
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        if backend in SUPPORTED_TVM_BACKENDS:
            model = ret["target_model"]
            if self.enable_vext:
                if "zvl" not in model:
                    model = f"{model}-zvl{self.vlen}b"
            ret["target_model"] = model
        return ret
