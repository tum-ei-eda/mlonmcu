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
"""Flow module for frameworks and backend."""

from mlonmcu.flow.tflm.framework import TFLMFramework
from mlonmcu.flow.tvm.framework import TVMFramework
from mlonmcu.flow.iree.framework import IREEFramework

from mlonmcu.flow.tflm.backend.tflmc import TFLMCBackend
from mlonmcu.flow.tflm.backend.tflmi import TFLMIBackend

from mlonmcu.flow.tvm.backend.tvmaot import TVMAOTBackend
from mlonmcu.flow.tvm.backend.tvmaotplus import TVMAOTPlusBackend
from mlonmcu.flow.tvm.backend.tvmcg import TVMCGBackend
from mlonmcu.flow.tvm.backend.tvmllvm import TVMLLVMBackend
from mlonmcu.flow.tvm.backend.tvmrt import TVMRTBackend

from mlonmcu.flow.iree.backend.ireevmvx import IREEVMVXBackend
from mlonmcu.flow.iree.backend.ireevmvx_inline import IREEVMVXInlineBackend
from mlonmcu.flow.iree.backend.ireellvm import IREELLVMBackend
from mlonmcu.flow.iree.backend.ireellvm_inline import IREELLVMInlineBackend
from mlonmcu.flow.iree.backend.ireellvmc import IREELLVMCBackend
from mlonmcu.flow.iree.backend.ireellvmc_inline import IREELLVMCInlineBackend

SUPPORTED_FRAMEWORKS = {
    "tflm": TFLMFramework,
    "tvm": TVMFramework,
    "iree": IREEFramework,
}

SUPPORTED_TFLITE_BACKENDS = {
    "tflmc": TFLMCBackend,
    "tflmi": TFLMIBackend,
}

SUPPORTED_TVM_BACKENDS = {
    "tvmaot": TVMAOTBackend,
    "tvmaotplus": TVMAOTPlusBackend,
    "tvmrt": TVMRTBackend,
    "tvmcg": TVMCGBackend,
    "tvmllvm": TVMLLVMBackend,
}

SUPPORTED_IREE_LLVM_BACKENDS = {
    "ireellvm": IREELLVMBackend,
    "ireellvm_inline": IREELLVMInlineBackend,
    "ireellvmc": IREELLVMCBackend,
    "ireellvmc_inline": IREELLVMCInlineBackend,
}

SUPPORTED_IREE_BACKENDS = {
    "ireevmvx": IREEVMVXBackend,
    "ireevmvx_inline": IREEVMVXInlineBackend,
    **SUPPORTED_IREE_LLVM_BACKENDS,
}

SUPPORTED_FRAMEWORK_BACKENDS = {
    "tflm": SUPPORTED_TFLITE_BACKENDS,
    "tvm": SUPPORTED_TVM_BACKENDS,
    "iree": SUPPORTED_IREE_BACKENDS,
}

SUPPORTED_BACKENDS = {**SUPPORTED_TFLITE_BACKENDS, **SUPPORTED_TVM_BACKENDS, **SUPPORTED_IREE_BACKENDS}


def get_available_backend_names():
    """Return all available backend names."""
    return list(SUPPORTED_BACKENDS.keys())
