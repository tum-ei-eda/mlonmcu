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

### from mlonmcu.flow.tflm.framework import TFLMFramework
### from mlonmcu.flow.tvm.framework import TVMFramework
### from mlonmcu.flow.iree.framework import IREEFramework
### from mlonmcu.flow.emx.framework import EMXFramework
### from mlonmcu.flow.executorch.framework import ExecutorchFramework
###
### # from mlonmcu.flow.none.framework import NoneFramework
###
### from mlonmcu.flow.tflm.backend.tflmc import TFLMCBackend
### from mlonmcu.flow.tflm.backend.tflmi import TFLMIBackend
###
### from mlonmcu.flow.tvm.backend.tvmaot import TVMAOTBackend
### from mlonmcu.flow.tvm.backend.tvmaotplus import TVMAOTPlusBackend
### from mlonmcu.flow.tvm.backend.tvmcg import TVMCGBackend
### from mlonmcu.flow.tvm.backend.tvmllvm import TVMLLVMBackend
### from mlonmcu.flow.tvm.backend.tvmrt import TVMRTBackend
###
### from mlonmcu.flow.iree.backend.ireevmvx import IREEVMVXBackend
### from mlonmcu.flow.iree.backend.ireevmvx_inline import IREEVMVXInlineBackend
### from mlonmcu.flow.iree.backend.ireellvm import IREELLVMBackend
### from mlonmcu.flow.iree.backend.ireellvm_inline import IREELLVMInlineBackend
### from mlonmcu.flow.iree.backend.ireellvmc import IREELLVMCBackend
### from mlonmcu.flow.iree.backend.ireellvmc_inline import IREELLVMCInlineBackend
###
### from mlonmcu.flow.emx.backend import EMXBackend
### from mlonmcu.flow.executorch.backend import ExecutorchBackend

# from mlonmcu.flow.none.backend.none import NoneBackend
from .registry import SUPPORTED_TVM_BACKENDS, SUPPORTED_FRAMEWORKS, SUPPORTED_BACKENDS, get_available_backend_names
from .framework import Framework
from .backend import Backend

__all__ = ["SUPPORTED_TVM_BACKENDS", "SUPPORTED_FRAMEWORKS", "SUPPORTED_BACKENDS", "Framework", "Backend", "get_available_backend_names"]
