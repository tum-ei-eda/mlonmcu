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
"""TVM Build Platform"""
from ..platform import BuildPlatform
from .tvm_backend import create_tvm_platform_backend, get_tvm_platform_backends


class TvmBuildPlatform(BuildPlatform):
    """TVM build platform class."""

    def create_backend(self, name):
        supported = self.get_supported_backends()
        assert name in supported, f"{name} is not a valid TVM platform backend"
        base = supported[name]
        return create_tvm_platform_backend(name, self, base=base)

    def get_supported_backends(self):
        backend_names = get_tvm_platform_backends()
        return backend_names
