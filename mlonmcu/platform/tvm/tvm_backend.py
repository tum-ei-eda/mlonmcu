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
from mlonmcu.flow.tvm.backend.tvmllvm import TVMLLVMBackend

logger = get_logger()

TVM_PLATFORM_BACKEND_REGISTRY = {}


def register_tvm_platform_backend(backend_name, b, override=False):
    global TVM_PLATFORM_BACKEND_REGISTRY

    if backend_name in TVM_PLATFORM_BACKEND_REGISTRY and not override:
        raise RuntimeError(f"TVM platform backend  {backend_name} is already registered")
    TVM_PLATFORM_BACKEND_REGISTRY[backend_name] = b


def get_tvm_platform_backends():
    return TVM_PLATFORM_BACKEND_REGISTRY


register_tvm_platform_backend("tvmllvm", TVMLLVMBackend)


def create_tvm_platform_backend(name, platform, base=TVMBackend):
    class TvmPlatformBackend(base):
        def __init__(self, features=None, config=None):
            super().__init__(runtime="cpp", fmt="so", features=features, config=config)
            self.platform = platform

    return TvmPlatformBackend
