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
"""Definitions for EMXFramework."""

from pathlib import Path

from mlonmcu.flow.framework import Framework


class EMXFramework(Framework):
    """EMX Framework specialization."""

    name = "emx"

    FEATURES = set()

    DEFAULTS = {}

    REQUIRED = {"emx.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)

    @property
    def emx_src_dir(self):
        return Path(self.config["emx.src_dir"])

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["EMX_SRC_DIR"] = str(self.emx_src_dir)
        return ret
