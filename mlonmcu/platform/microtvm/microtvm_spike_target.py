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
from pathlib import Path

from mlonmcu.target.riscv.riscv_pext_target import RVPTarget
from mlonmcu.target.riscv.riscv_vext_target import RVVTarget
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.riscv.util import update_extensions

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class SpikeMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget, RVPTarget, RVVTarget):
    FEATURES = TemplateMicroTvmPlatformTarget.FEATURES | RVPTarget.FEATURES | RVVTarget.FEATURES

    DEFAULTS = {
        **TemplateMicroTvmPlatformTarget.DEFAULTS,
        **RVPTarget.DEFAULTS,
        **RVVTarget.DEFAULTS,
        "verbose": False,
        "quiet": True,
        "workspace_size_bytes": None,
        "toolchain": "gcc",
    }
    REQUIRED = (
        TemplateMicroTvmPlatformTarget.REQUIRED
        | RVPTarget.REQUIRED
        | RVVTarget.REQUIRED
        | {
            "llvm.install_dir",
            "spike.exe",
            "spike.pk",
            "microtvm_spike.src_dir",
        }
    )

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.microtvm_spike_src_dir / "template_project"
        # TODO: interate into TVM build config
        self.option_names = [
            "verbose",
            "quiet",
            "workspace_size_bytes",
            # TODO
        ]

    @property
    def microtvm_spike_src_dir(self):
        return Path(self.config["microtvm_spike.src_dir"])

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def spike_pk(self):
        return Path(self.config["spike.pk"])

    @property
    def llvm_prefix(self):
        return Path(self.config["llvm.install_dir"])

    @property
    def toolchain(self):
        value = self.config["toolchain"]
        assert value in ["gcc", "llvm"]
        return value

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "gcc_prefix": self.riscv_gcc_prefix,
                "gcc_name": self.riscv_gcc_basename,
                "llvm_dir": self.llvm_prefix,
                "toolchain": self.toolchain,
                "spike_exe": str(self.spike_exe),
                "spike_pk": str(self.spike_pk),
                "arch": self.arch,
                "abi": self.abi,
                "vlen": self.vlen,
                "elen": self.elen,
            }
        )
        return ret

    @property
    def extensions(self):
        # exts = RVPTarget.extensions(self) + RVVTarget.extensions(self)
        exts = super().extensions
        return update_extensions(
            exts,
        )

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        model = ret["target_model"]
        _, arch = model.split("-", 1)
        ret["target_model"] = f"spike-{arch}"
        if backend in SUPPORTED_TVM_BACKENDS:
            if optimized_layouts:
                if self.enable_pext or self.enable_vext:
                    ret.update(
                        {
                            # Warning: passing kernel layouts does not work with upstream TVM
                            # TODO: allow passing map?
                            "desired_layout": "NHWC:HWOI",
                        }
                    )
        return ret
