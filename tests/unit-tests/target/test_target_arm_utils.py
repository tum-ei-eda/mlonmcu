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
import pytest

from mlonmcu.target.arm.util import resolve_cpu_features


@pytest.mark.parametrize("model", ["cortex-m0", "cortex-m0plus", "cortex-m1", "cortex-m3", "cortex-m23"])
def test_target_arm_resolve_cpu_features1(model):
    # m0, m0plus, m1, m3, m23
    DSP = False
    MVE = False
    FP = False
    FP_DP = False
    assert resolve_cpu_features(model, None, None, None, None) == (model, "soft", "auto")
    assert resolve_cpu_features(model, FP, FP_DP, DSP, MVE) == (model, "soft", "auto")
    # assert resolve_cpu_features("cortex-m0", enable_fp, enable_fp_dp, enable_dsp, enable_mve) == (cpu, abi, fpu)


@pytest.mark.parametrize("model", ["cortex-m4"])
def test_target_arm_resolve_cpu_features2(model):
    DSP = True
    MVE = False
    FP = True
    FP_DP = False
    assert resolve_cpu_features(model, None, None, None, None) == (model, "hard", "auto")
    assert resolve_cpu_features(model + "+nofp", None, None, None, None) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model, FP, FP_DP, DSP, MVE) == (model, "hard", "auto")
    assert resolve_cpu_features(model, False, FP_DP, DSP, MVE) == (model + "+nofp", "soft", "auto")


@pytest.mark.parametrize("model", ["cortex-m33", "cortex-m35p"])
def test_target_arm_resolve_cpu_features3(model):
    DSP = True
    MVE = False
    FP = True
    FP_DP = False
    assert resolve_cpu_features(model, None, None, None, None) == (model, "hard", "auto")
    assert resolve_cpu_features(model + "+nofp", None, None, None, None) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model + "+nodsp", None, None, None, None) == (model + "+nodsp", "hard", "auto")
    assert resolve_cpu_features(model + "+nofp+nodsp", None, None, None, None) == (
        model + "+nofp+nodsp",
        "soft",
        "auto",
    )
    assert resolve_cpu_features(model, FP, FP_DP, DSP, MVE) == (model, "hard", "auto")
    assert resolve_cpu_features(model, False, FP_DP, DSP, MVE) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model, FP, FP_DP, False, MVE) == (model + "+nodsp", "hard", "auto")
    assert resolve_cpu_features(model, False, FP_DP, False, MVE) == (model + "+nofp+nodsp", "soft", "auto")


@pytest.mark.parametrize("model", ["cortex-m7"])
def test_target_arm_resolve_cpu_features4(model):
    DSP = True
    MVE = False
    FP = True
    FP_DP = True
    assert resolve_cpu_features(model, None, None, None, None) == (model, "hard", "auto")
    assert resolve_cpu_features(model + "+nofp", None, None, None, None) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model + "+nofp.dp", None, None, None, None) == (model + "+nofp.dp", "hard", "auto")
    assert resolve_cpu_features(model + "+nofp+nofp.dp", None, None, None, None) == (
        model + "+nofp+nofp.dp",
        "soft",
        "auto",
    )
    assert resolve_cpu_features(model, FP, FP_DP, DSP, MVE) == (model, "hard", "auto")
    assert resolve_cpu_features(model, False, False, DSP, MVE) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model, FP, False, DSP, MVE) == (model + "+nofp.dp", "hard", "auto")


@pytest.mark.parametrize("model", ["cortex-m55"])
def test_target_arm_resolve_cpu_features5(model):
    DSP = True
    MVE = True
    FP = True
    FP_DP = True
    assert resolve_cpu_features(model, None, None, None, None) == (model, "hard", "auto")
    assert resolve_cpu_features(model + "+nofp", None, None, None, None) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model + "+nodsp+nomve", None, None, None, None) == (
        model + "+nodsp+nomve",
        "hard",
        "auto",
    )
    assert resolve_cpu_features(model + "+nofp+nodsp+nomve", None, None, None, None) == (
        model + "+nofp+nodsp+nomve",
        "soft",
        "auto",
    )
    assert resolve_cpu_features(model + "+nomve", None, None, None, None) == (model + "+nomve", "hard", "auto")
    assert resolve_cpu_features(model, FP, FP_DP, DSP, MVE) == (model, "hard", "auto")
    assert resolve_cpu_features(model, False, False, DSP, MVE) == (model + "+nofp", "soft", "auto")
    assert resolve_cpu_features(model, FP, FP_DP, False, False) == (model + "+nodsp+nomve", "hard", "auto")
    assert resolve_cpu_features(model, False, False, False, False) == (model + "+nofp+nodsp+nomve", "soft", "auto")
    assert resolve_cpu_features(model, FP, FP_DP, DSP, False) == (model + "+nomve", "hard", "auto")
