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

from mlonmcu.target.target import Target

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class GVSocMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    # FEATURES = TemplateMicroTvmPlatformTarget.FEATURES + ["xpulp"]
    FEATURES = TemplateMicroTvmPlatformTarget.FEATURES

    DEFAULTS = {
        **TemplateMicroTvmPlatformTarget.DEFAULTS,
        # "verbose": True,
        "compiler": "gcc",
        "project_type": "host_driven",
        # "xpulp_version": None,  # None means that xpulp extension is not used,
        # "model": "pulp",
    }
    REQUIRED = Target.REQUIRED | {
        "gvsoc.exe",
        "pulp_freertos.support_dir",
        "pulp_freertos.config_dir",
        "pulp_freertos.install_dir",
        "microtvm_gvsoc.template",
        "hannah_tvm.src_dir",
    }

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.microtvm_gvsoc_template
        # TODO: integrate into TVM build config
        self.option_names = [
            # "verbose",
            "project_type",
            "compiler",
        ]

    @property
    def microtvm_gvsoc_template(self):
        return Path(self.config["microtvm_gvsoc.template"])

    @property
    def hannah_tvm_src_dir(self):
        return Path(self.config["hannah_tvm.src_dir"])

    @property
    def compiler(self):
        return self.config["compiler"]

    def get_project_options(self):
        ret = super().get_project_options()
        # TODO
        ret.update(
            {
                # "gvsoc_exe": str(self.gvsoc_exe),
            }
        )
        return ret

    def update_environment(self, env):
        super().update_environment(env)
        p = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.hannah_tvm_src_dir}:{p}"

        # TODO
        # if "PATH" in env:
        #     env["PATH"] = str(self.riscv_gcc_install_dir / "bin") + ":" + env["PATH"]
        # else:
        #     env["PATH"] = str(self.riscv_gcc_install_dir / "bin")

    def get_backend_config(self, backend):
        ret = {}
        # TODO
        return ret
