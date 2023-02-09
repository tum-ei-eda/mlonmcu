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
"""MLonMCU ARM Cortex-M utilities"""

DSP_TARGETS = ["cortex-m4", "cortex-m7", "cortex-m33", "cortex-m35p", "cortex-m55"]

MVE_TARGETS = ["cortex-m55"]

FP_TARGETS = ["cortex-m4", "cortex-m7", "cortex-m33", "cortex-m35p", "cortex-m55"]
FP_DP_TARGETS = ["cortex-m7", "cortex-m55"]

ALLOWED_ATTRS = {
    "cortex-m0": [],
    "cortex-m0plus": [],
    "cortex-m1": [],
    "cortex-m3": [],
    "cortex-m4": ["+nofp"],
    "cortex-m7": ["+nofp", "+nofp.dp"],
    "cortex-m23": [],
    "cortex-m33": ["+nodsp", "+nofp"],
    "cortex-m35p": ["+nodsp", "+nofp"],
    "cortex-m55": ["+nomve", "+nomve.fp", "+nodsp", "+nofp"],
}


def resolve_cpu_features(model, enable_fp=None, enable_fp_dp=None, enable_dsp=None, enable_mve=None):
    cpu = model
    fpu = "auto"

    splitted = cpu.split("+")
    base = splitted[0]
    attrs = [f"+{attr}" for attr in splitted[1:]]

    if enable_fp is None:
        enable_fp = base in FP_TARGETS and ("+nofp" not in ALLOWED_ATTRS[base] or "+nofp" not in attrs)
    if enable_fp_dp is None:
        if enable_fp:
            enable_fp_dp = base in FP_DP_TARGETS and ("+nofp.dp" not in ALLOWED_ATTRS[base] or "+nofp.dp" not in attrs)
        else:
            enable_fp_dp = False
    if enable_dsp is None:
        enable_dsp = base in DSP_TARGETS and ("+nodsp" not in ALLOWED_ATTRS[base] or "+nodsp" not in attrs)
    if enable_mve is None:
        enable_mve = base in MVE_TARGETS and ("+nomve" not in ALLOWED_ATTRS[base] or "+nomve" not in attrs)

    float_abi = "hard" if enable_fp else "soft"

    if enable_fp_dp:
        assert enable_fp, "ARM double precision floating point supports requires fp extension"
    if enable_mve:
        assert enable_dsp, "ARM MVEI extension implies DSP extension"

    if len(attrs) > 0:
        # User defined features explicitly, check for validity
        if enable_fp and ("+nofp" not in ALLOWED_ATTRS[base] or "+nofp" not in attrs):
            assert base in FP_TARGETS, f"Chosen cpu '{cpu}' does not support floating point"
        if enable_fp_dp and ("+nofp.dp" not in ALLOWED_ATTRS[base] or "+nofp.dp" not in attrs):
            assert base in FP_DP_TARGETS, f"Chosen cpu '{cpu}' does not support double precison floating point"
        if enable_dsp and ("+nodsp" not in ALLOWED_ATTRS[base] or "+nodsp" not in attrs):
            assert base in DSP_TARGETS, f"Chosen cpu '{cpu}' does not support dsp instructions"
        if enable_mve and ("+nomve" not in ALLOWED_ATTRS[base] or "+nomve" not in attrs):
            assert base in MVE_TARGETS, f"Chosen cpu '{cpu}' does not support mve (helium) instructions"
    else:
        # Define features automatically
        if enable_fp:
            assert base in FP_TARGETS, f"Chosen cpu '{cpu}' does not support floating point"
        else:
            if base in FP_TARGETS:
                assert "+nofp" in ALLOWED_ATTRS[base], f"Floating point support can not be turned off for cpu '{cpu}'"
                attrs.append("+nofp")
        if enable_fp_dp:
            assert base in FP_DP_TARGETS, f"Chosen cpu '{cpu}' does not support double precision floating point"
        else:
            if base in FP_DP_TARGETS and enable_fp:
                assert (
                    "+nofp.dp" in ALLOWED_ATTRS[base]
                ), f"Double precision floating point support can not be turned off for cpu '{cpu}'"
                attrs.append("+nofp.dp")
        if enable_dsp:
            assert base in DSP_TARGETS, f"Chosen cpu '{cpu}' does not support dsp instructions"
        else:
            if base in DSP_TARGETS:
                assert "+nodsp" in ALLOWED_ATTRS[base], f"DSP extension can not be turned off for cpu '{cpu}'"
                attrs.append("+nodsp")
        if enable_mve:
            assert base in MVE_TARGETS, f"Chosen cpu '{cpu}' does not support mve (helium) instructions"
        else:
            if base in MVE_TARGETS:
                assert "+nomve" in ALLOWED_ATTRS[base], f"MVE (helium) extension can not be turned of for cpu '{cpu}'"
                attrs.append("+nomve")

    cpu = "".join([base] + attrs)
    return cpu, float_abi, fpu
