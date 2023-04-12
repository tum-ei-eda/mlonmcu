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

import os
import pkg_resources
from pathlib import Path

from mlonmcu.flow.framework import Framework

# from mlonmcu.flow.tvm import TVMBackend


def get_crt_config_dir():
    files = pkg_resources.resource_listdir(
        "mlonmcu", os.path.join("..", "resources", "frameworks", "tvm", "crt_config")
    )
    if "crt_config.h" not in files:
        return None
    fname = pkg_resources.resource_filename(
        "mlonmcu", os.path.join("..", "resources", "frameworks", "tvm", "crt_config")
    )
    return fname


class TVMFramework(Framework):
    """TVM Framework specialization."""

    name = "tvm"

    FEATURES = {"cmsisnnbyoc", "muriscvnnbyoc"}

    DEFAULTS = {
        "extra_incs": [],
        "extra_libs": [],
        "crt_config_dir": get_crt_config_dir(),
    }

    REQUIRED = {"tvm.src_dir"}

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

    @property
    def crt_config_dir(self):
        return self.config["crt_config_dir"]

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.extra_incs or self.extra_libs:
            assert platform == "mlif", "Extra incs or libs are only supported by 'mlif' platform"
            if self.extra_incs:
                if isinstance(self.extra_incs, list):
                    temp = r"\;".join(self.extra_incs)
                else:
                    temp = self.extra_incs
                ret["TVM_EXTRA_INCS"] = temp
            if self.extra_libs:
                if isinstance(self.extra_libs, list):
                    temp = r"\;".join(self.extra_libs)
                else:
                    temp = self.extra_libs
                ret["TVM_EXTRA_LIBS"] = temp
        if self.crt_config_dir:
            ret["TVM_CRT_CONFIG_DIR"] = self.crt_config_dir
        ret["TVM_DIR"] = str(self.tvm_src)
        return ret
