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
"""Registry for frameworks and backend."""

from mlonmcu.flow.tflm.framework import TFLMFramework
from mlonmcu.flow.tvm.framework import TVMFramework
from mlonmcu.flow.iree.framework import IREEFramework
from mlonmcu.flow.emx.framework import EMXFramework
from mlonmcu.flow.executorch.framework import ExecutorchFramework

from mlonmcu.flow.none import NoneFramework, NoneBackend

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

from mlonmcu.flow.emx.backend import EMXBackend
from mlonmcu.flow.executorch.backend import ExecutorchBackend

SUPPORTED_FRAMEWORKS = {}
SUPPORTED_BACKENDS = {}
SUPPORTED_FRAMEWORK_BACKENDS = {}
SUPPORTED_TFLITE_BACKENDS = {}
SUPPORTED_TVM_BACKENDS = {}
SUPPORTED_IREE_LLVM_BACKENDS = {}
SUPPORTED_IREE_BACKENDS = {}
SUPPORTED_EMX_BACKENDS = {}
SUPPORTED_EXECUTORCH_BACKENDS = {}
SUPPORTED_NONE_BACKENDS = {}


def register_framework(framework_name, framework_cls=None, override=False):
    """Register a framework class under the given name."""

    def _register(cls):
        if framework_name in SUPPORTED_FRAMEWORKS and not override:
            raise RuntimeError(f"Framework {framework_name} is already registered")
        SUPPORTED_FRAMEWORKS[framework_name] = cls
        return cls

    if framework_cls is None:
        return _register
    return _register(framework_cls)


def register_backend(backend_name, backend_cls=None, framework=None, override=False):
    """Register a backend class under the given name."""

    def _register(cls):
        if backend_name in SUPPORTED_BACKENDS and not override:
            raise RuntimeError(f"Backend {backend_name} is already registered")
        SUPPORTED_BACKENDS[backend_name] = cls
        if framework:
            if framework not in SUPPORTED_FRAMEWORK_BACKENDS:
                SUPPORTED_FRAMEWORK_BACKENDS[framework] = {}
            SUPPORTED_FRAMEWORK_BACKENDS[framework][backend_name] = cls
        return cls

    if backend_cls is None:
        return _register
    return _register(backend_cls)


def get_frameworks():
    """Return registered frameworks."""
    return SUPPORTED_FRAMEWORKS


def get_backends(framework=None):
    """Return registered backends, optionally filtered by framework."""
    if framework is None:
        return SUPPORTED_BACKENDS
    return SUPPORTED_FRAMEWORK_BACKENDS.get(framework, {})


def get_available_backend_names():
    """Return all available backend names."""
    return list(SUPPORTED_BACKENDS.keys())


register_framework("tflm", TFLMFramework)
register_framework("tvm", TVMFramework)
register_framework("iree", IREEFramework)
register_framework("emx", EMXFramework)
register_framework("executorch", ExecutorchFramework)
register_framework("none", NoneFramework)

register_backend("tflmc", TFLMCBackend, framework="tflm")
register_backend("tflmi", TFLMIBackend, framework="tflm")
SUPPORTED_TFLITE_BACKENDS.update(get_backends("tflm"))

register_backend("tvmaot", TVMAOTBackend, framework="tvm")
register_backend("tvmaotplus", TVMAOTPlusBackend, framework="tvm")
register_backend("tvmrt", TVMRTBackend, framework="tvm")
register_backend("tvmcg", TVMCGBackend, framework="tvm")
register_backend("tvmllvm", TVMLLVMBackend, framework="tvm")
SUPPORTED_TVM_BACKENDS.update(get_backends("tvm"))

register_backend("ireevmvx", IREEVMVXBackend, framework="iree")
register_backend("ireevmvx_inline", IREEVMVXInlineBackend, framework="iree")
register_backend("ireellvm", IREELLVMBackend, framework="iree")
register_backend("ireellvm_inline", IREELLVMInlineBackend, framework="iree")
register_backend("ireellvmc", IREELLVMCBackend, framework="iree")
register_backend("ireellvmc_inline", IREELLVMCInlineBackend, framework="iree")
SUPPORTED_IREE_BACKENDS.update(get_backends("iree"))
SUPPORTED_IREE_LLVM_BACKENDS.update(
    {
        "ireellvm": IREELLVMBackend,
        "ireellvm_inline": IREELLVMInlineBackend,
        "ireellvmc": IREELLVMCBackend,
        "ireellvmc_inline": IREELLVMCInlineBackend,
    }
)

register_backend("emx", EMXBackend, framework="emx")
SUPPORTED_EMX_BACKENDS.update(get_backends("emx"))

register_backend("executorch", ExecutorchBackend, framework="executorch")
SUPPORTED_EXECUTORCH_BACKENDS.update(get_backends("executorch"))

register_backend("none", NoneBackend, framework="none")
SUPPORTED_NONE_BACKENDS.update(get_backends("none"))
