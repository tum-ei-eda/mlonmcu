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
from mlonmcu.logging import get_logger

from mlonmcu.flow.tvm.backend.backend import TVMBackend

from mlonmcu.flow.tvm.backend.tvmaot import TVMAOTBackend

# from mlonmcu.flow.tvm.backend.tvmaotplus import TVMAOTPlusBackend
from mlonmcu.flow.tvm.backend.tvmllvm import TVMLLVMBackend
from mlonmcu.flow.tvm.backend.tvmrt import TVMRTBackend

logger = get_logger()

MICROTVM_PLATFORM_BACKEND_REGISTRY = {}


def register_microtvm_platform_backend(backend_name, b, override=False):
    global MICROTVM_PLATFORM_BACKEND_REGISTRY

    if backend_name in MICROTVM_PLATFORM_BACKEND_REGISTRY and not override:
        raise RuntimeError(f"TVM platform backend  {backend_name} is already registered")
    MICROTVM_PLATFORM_BACKEND_REGISTRY[backend_name] = b


def get_microtvm_platform_backends():
    return MICROTVM_PLATFORM_BACKEND_REGISTRY


register_microtvm_platform_backend("tvmaot", TVMAOTBackend)
# register_microtvm_platform_backend("tvmaotplus", TVMAOTPlusBackend)
register_microtvm_platform_backend("tvmrt", TVMRTBackend)
register_microtvm_platform_backend("tvmllvm", TVMLLVMBackend)


def create_microtvm_platform_backend(name, platform, base=TVMBackend):
    class MicroTvmPlatformBackend(base):
        def __init__(self, features=None, config=None):
            super().__init__(runtime="crt", fmt="mlf", system_lib=True, features=features, config=config)
            self.platform = platform

    return MicroTvmPlatformBackend
