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
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.config import str2bool
import mlonmcu.target.riscv.util as riscv_util

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class SpikeMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    FEATURES = TemplateMicroTvmPlatformTarget.FEATURES + ["vext", "pext"]

    DEFAULTS = {
        **TemplateMicroTvmPlatformTarget.DEFAULTS,
        "verbose": True,
        "arch": "rv32gc",
        "abi": None,
        "spike_extra_args": None,
        "pk_extra_args": None,
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
        "enable_pext": False,
        "pext_spec": 0.92,  # ?
        "vlen": 0,  # vectorization=off
        "elen": 32,
    }
    REQUIRED = Target.REQUIRED + [
        "spike.exe",
        "spike.pk",
        "riscv_gcc.name",
        "riscv_gcc.install_dir",
        "riscv_gcc.variant",
        "tvm.src_dir",
    ]

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.tvm_src_dir / "apps" / "microtvm" / "spike"
        # TODO: interate into TVM build config
        self.option_names = [
            "verbose",
            "spike_exe",
            "spike_pk",
            "arch",
            "abi",
            "triple",
            "spike_extra_args",
            "pk_extra_args",
        ]

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def spike_pk(self):
        return Path(self.config["spike.pk"])

    @property
    def riscv_gcc_name(self):
        return self.config["riscv_gcc.name"]

    @property
    def gcc_variant(self):
        return self.config["riscv_gcc.variant"]

    @property
    def riscv_gcc_install_dir(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def tvm_src_dir(self):
        return Path(self.config["tvm.src_dir"])

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def fpu(self):
        exts = self.extensions
        if "g" in exts or "d" in exts:
            return "double"
        elif "f" in exts:
            return "single"
        return "none"

    @property
    def has_fpu(self):
        return self.fpu != "none"

    @property
    def xlen(self):
        arch = self.config["arch"]
        xlen = int(arch[2:4])
        return xlen

    @property
    def extensions(self):
        arch = self.config["arch"]
        exts_str = arch[4:]

        def _split_exts(x):
            ret = []
            splitted = x.split("_")
            for c in splitted:
                lowered = c.lower()
                assert len(c) > 0
                if len(c) == 1:
                    ret.append(lowered)
                else:
                    for i, cc in enumerate(lowered):
                        if cc in ["x", "z"]:
                            ret.append(lowered[i:])
                            break
                        else:
                            ret.append(cc)
            return ret

        exts = _split_exts(exts_str)
        if "g" in exts or "d" in exts:
            fpu = "double"
        elif "f" in exts:
            fpu = "single"
        return riscv_util.update_extensions(
            exts,
            pext=self.enable_pext,
            pext_spec=self.pext_spec,
            vext=self.enable_vext,
            elen=self.elen,
            embedded=self.embedded_vext,
            fpu=fpu,
            variant=self.gcc_variant,
        )

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def elen(self):
        return int(self.config["elen"])

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def pext_spec(self):
        return float(self.config["pext_spec"])

    @property
    def arch(self):
        exts_str = riscv_util.join_extensions(riscv_util.sort_extensions_canonical(self.extensions, lower=True))
        return f"rv{self.xlen}{exts_str}"

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "spike_exe": str(self.spike_exe),
                "spike_pk": str(self.spike_pk),
                "triple": str(self.riscv_gcc_install_dir / "bin" / self.riscv_gcc_name),
            }
        )
        return ret

    def update_environment(self, env):
        super().update_environment(env)
        if "PATH" in env:
            env["PATH"] = str(self.riscv_gcc_install_dir / "bin") + ":" + env["PATH"]
        else:
            env["PATH"] = str(self.riscv_gcc_install_dir / "bin")

    def get_backend_config(self, backend):
        ret = {}
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update(
                {
                    "target_device": "riscv_cpu",
                    "target_march": self.arch,
                    "target_model": f"spike-{self.arch}",
                    "target_mtriple": self.riscv_gcc_name,
                    "target_mabi": self.config.get("abi", None),
                }
            )
            if self.enable_pext or self.enable_vext:
                ret.update(
                    {
                        # Warning: passing kernel layouts does not work with upstream TVM
                        # TODO: allow passing map?
                        "desired_layout": "NHWC:HWOI",
                    }
                )
        return ret
