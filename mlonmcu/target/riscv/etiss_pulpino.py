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
"""MLonMCU ETISS/Pulpino Target definitions"""


from mlonmcu.logging import get_logger
from mlonmcu.target.common import cli
from .etiss import EtissTarget

logger = get_logger()


class EtissPulpinoTarget(EtissTarget):
    """Target using a Pulpino-like VP running in the ETISS simulator"""

    REQUIRED = EtissTarget.REQUIRED | {"etiss.src_dir"}

    def __init__(self, name="etiss_pulpino", features=None, config=None):
        super().__init__(name, features=features, config=config)

    def get_ini_bool_config(self, override=None):
        ret = super().get_ini_bool_config(override=override)
        ret["arch.enable_semihosting"] = False
        return ret

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        ret = super().get_platform_defs(platform)
        ret["ETISS_DIR"] = self.etiss_dir
        del ret["MEM_ROM_ORIGIN"]
        del ret["MEM_ROM_LENGTH"]
        del ret["MEM_RAM_ORIGIN"]
        del ret["MEM_RAM_LENGTH"]
        ret["PULPINO_ROM_START"] = self.rom_start
        ret["PULPINO_ROM_SIZE"] = self.rom_size
        ret["PULPINO_RAM_START"] = self.ram_start
        ret["PULPINO_RAM_SIZE"] = self.ram_size
        return ret


if __name__ == "__main__":
    cli(target=EtissPulpinoTarget)
