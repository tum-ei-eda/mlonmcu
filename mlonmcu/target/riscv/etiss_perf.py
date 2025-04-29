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

from pathlib import Path

from mlonmcu.logging import get_logger
from .etiss import EtissTarget

logger = get_logger()


class EtissPerfTarget(EtissTarget):
    """Target using a simple RISC-V VP running in the ETISS simulator"""

    FEATURES = EtissTarget.FEATURES | {
        "perf_sim",
    }

    DEFAULTS = {
        **EtissTarget.DEFAULTS,
        "use_run_helper": False,
    }
    REQUIRED = EtissTarget.REQUIRED | {
        "etiss_perf.src_dir",
        "etiss_perf.install_dir",
        "etiss_perf.exe",
    }

    def __init__(self, name="etiss_perf", features=None, config=None):
        super().__init__(name, features=features, config=config)
        # TODO: make optional or move to mlonmcu pkg
        self.metrics_script = Path(self.etiss_src_dir) / "src" / "bare_etiss_processor" / "get_metrics.py"

    @property
    def etiss_src_dir(self):
        return self.config["etiss_perf.src_dir"]

    @property
    def etiss_dir(self):
        return self.config["etiss_perf.install_dir"]

    @property
    def etiss_script(self):
        assert not self.use_run_helper, "Target etiss_perf does not support run_helper.sh"

    @property
    def etiss_exe(self):
        return self.config["etiss_perf.exe"]
