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
"""Definitions for ExecutorchFramework."""

from pathlib import Path

from mlonmcu.flow.framework import Framework


class ExecutorchFramework(Framework):
    """Executorch Framework specialization."""

    name = "executorch"

    FEATURES = set()

    DEFAULTS = {}

    REQUIRED = {"executorch.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)

    @property
    def executorch(self):
        return Path(self.config["executorch.src_dir"])

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["EXECUTORCH_SRC_DIR"] = str(self.executorch_src_dir)
        return ret
