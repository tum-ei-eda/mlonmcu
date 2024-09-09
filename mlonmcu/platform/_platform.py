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
from .mlif import MlifPlatform
from .espidf import EspIdfPlatform
from .zephyr import ZephyrPlatform
from .tvm import TvmPlatform
from .microtvm import MicroTvmPlatform

# from .arduino import ArduinoPlatform


PLATFORM_REGISTRY = {}


def register_platform(platform_name, p, override=False):
    global PLATFORM_REGISTRY

    if platform_name in PLATFORM_REGISTRY and not override:
        raise RuntimeError(f"Platform {platform_name} is already registered")
    PLATFORM_REGISTRY[platform_name] = p


def get_platforms():
    return PLATFORM_REGISTRY


register_platform("mlif", MlifPlatform)
register_platform("espidf", EspIdfPlatform)
register_platform("zephyr", ZephyrPlatform)
register_platform("tvm", TvmPlatform)
register_platform("microtvm", MicroTvmPlatform)
