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
"""Definitions for TVMFramework."""

from pathlib import Path

from mlonmcu.flow.framework import Framework

# from mlonmcu.flow.tvm import TVMBackend


class TVMFramework(Framework):
    """TVM Framework specialization."""

    name = "tvm"

    FEATURES = ["cmsisnnbyoc"]

    DEFAULTS = {
        "extra_incs": [],
        "extra_libs": [],
    }

    REQUIRED = ["tvm.src_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)
        # self.backends = TVMBackend.registry

    @property
    def tvm_src(self):
        return Path(self.config["tvm.src_dir"])

    @property
    def extra_incs(self):
        return self.config["extra_incs"]

    @property
    def extra_libs(self):
        return self.config["extra_libs"]

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.extra_incs or self.extra_libs:
            assert platform == "mlif", "Extra incs or libs are only supported by 'mlif' platform"
            if self.extra_incs:
                temp = r"\;".join(self.extra_incs)
                ret["TVM_EXTRA_INCS"] = temp
            if self.extra_libs:
                temp = r"\;".join(self.extra_libs)
                ret["TVM_EXTRA_LIBS"] = temp
        ret["TVM_DIR"] = str(self.tvm_src)
        return ret
