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


class BackendModelOptions:
    def __init__(self, backend, supported=True, options={}):
        self.backend = backend
        self.supported = supported
        self.options = options


class TFLMIModelOptions(BackendModelOptions):
    def __init__(
        self,
        backend,
        supported=True,
        arena_size=None,
        builtin_ops=None,
        custom_ops=None,
    ):
        super().__init__(backend, supported=supported)
        self.arena_size = arena_size
        self.builtin_ops = builtin_ops
        self.custom_ops = custom_ops


class TVMRTModelOptions(BackendModelOptions):
    def __init__(self, backend, supported=True, arena_size=None):
        super().__init__(backend, supported=supported)
        self.arena_size = arena_size


def parse_model_options_for_backend(backend, options):
    backend_types = {
        "tflmi": TFLMIModelOptions,
        "tvmrt": TVMRTModelOptions,
    }
    if backend in backend_types:
        backend_type = backend_types[backend]
    else:
        backend_type = BackendModelOptions

    backend_options = backend_type(backend)

    for key, value in options.items():
        setattr(backend_options, key, value)

    return backend_options
