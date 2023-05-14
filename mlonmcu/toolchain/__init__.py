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
from .toolchain import (
    NoneToolchain,
    DefaultToolchain,
    GCCToolchain,
    LLVMToolchain,
    RISCVGCCToolchain,
    RISCVLLVMToolchain,
    RVVGCCToolchain,
    RVPGCCToolchain,
    XuantieRISCVGCCToolchain,
    PulpRISCVGCCToolchain,
    CoreVRISCVGCCToolchain,
    CoreVLLVMToolchain,
    ARMGCCToolchain,
    ARMLLVMToolchain,
)

SUPPORTED_TOOLCHAINS = {
    "none": NoneToolchain,  # For non-mlif targets
    "default": DefaultToolchain,  # Automatically pick via target system
    "gcc": GCCToolchain,
    "llvm": LLVMToolchain,
    "riscv_gcc": RISCVGCCToolchain,
    "riscv_llvm": RISCVLLVMToolchain,
    "riscv_gcc_vext": RVVGCCToolchain,
    "riscv_gcc_pext": RVPGCCToolchain,
    "xuantie_riscv_gcc": XuantieRISCVGCCToolchain,
    "pulp_riscv_gcc": PulpRISCVGCCToolchain,
    "corev_riscv_gcc": CoreVRISCVGCCToolchain,
    "corev_llvm": CoreVLLVMToolchain,
    "arm_gcc": ARMGCCToolchain,
    "arm_llvm": ARMLLVMToolchain,
}  # TODO: use registry instead

__all__ = []
