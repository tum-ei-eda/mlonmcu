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
"""Definitions for TFLMFramework."""

from pathlib import Path

from mlonmcu.flow.framework import Framework
from mlonmcu.flow.tflm import TFLMBackend
from mlonmcu.config import cfg, required, str2bool


class TFLMFramework(Framework):
    """TFLM Framework specialization."""

    name = "tflm"

    FEATURES = {"muriscvnn", "cmsisnn", "cfu_wca"}

    tf_src = required("tf.src_dir", cast=Path)
    optimized_kernel = cfg(None)
    optimized_kernel_inc_dirs = cfg([])
    optimized_kernel_libs = cfg([])
    override_dir = cfg(None)
    generate_tree = cfg(False, cast=str2bool)

    backends = TFLMBackend.registry

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)

    # @property
    # def cfu_accelerate(self):
    #     return str2bool(self.config["cfu_accelerate"], allow_none=True)

    @property
    def cfu_conv2d_idx_init(self):
        value = self.config["cfu_conv2d_idx_init"]
        if value is not None:
            value = int(value)
        return value

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.generate_tree:
            ret["TFLM_GENERATE_TREE"] = True
        if self.optimized_kernel or self.optimized_kernel_inc_dirs or self.optimized_kernel_libs:
            if self.optimized_kernel:
                ret["TFLM_OPTIMIZED_KERNEL"] = self.optimized_kernel
            if self.optimized_kernel_inc_dirs:
                if isinstance(self.optimized_kernel_inc_dirs, list):
                    temp = r"\;".join(self.optimized_kernel_inc_dirs)
                else:
                    temp = self.optimized_kernel_inc_dirs
                ret["TFLM_OPTIMIZED_KERNEL_INCLUDE_DIR"] = temp
            if self.optimized_kernel_libs:
                if isinstance(self.optimized_kernel_libs, list):
                    temp = r"\;".join(self.optimized_kernel_libs)
                else:
                    temp = self.optimized_kernel_libs
                ret["TFLM_OPTIMIZED_KERNEL_LIB"] = temp
        if platform in ["mlif", "mlif_litex"]:
            ret["TF_DIR"] = str(self.tf_src)
        # if self.cfu_accelerate:
        #     ret["CFU_ACCELERATE"] = True
        #     if self.cfu_conv2d_idx_init is not None:
        #         ret["CFU_CONV2D_IDX_INIT"] = self.cfu_conv2d_idx_init
        if self.override_dir:
            ret["TFLM_OVERRIDE"] = self.override_dir
        return ret
