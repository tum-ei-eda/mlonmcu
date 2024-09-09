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

from .common import get_task_factory

__all__ = ["get_task_factory"]

from .arm_gcc import *  # noqa: F401, F403
from .cmsisnn import *  # noqa: F401, F403
from .corstone300 import *  # noqa: F401, F403
from .espidf import *  # noqa: F401, F403
from .etiss import *  # noqa: F401, F403
from .llvm import *  # noqa: F401, F403
from .mlif import *  # noqa: F401, F403
from .muriscvnn import *  # noqa: F401, F403
from .riscv_gcc import *  # noqa: F401, F403
from .spike import *  # noqa: F401, F403
from .tflite import *  # noqa: F401, F403
from .tflmc import *  # noqa: F401, F403
from .tf import *  # noqa: F401, F403
from .tvm import *  # noqa: F401, F403
from .utvmcg import *  # noqa: F401, F403
from .zephyr import *  # noqa: F401, F403
from .pulp import *  # noqa: F401, F403
from .ekut import *  # noqa: F401, F403
from .ara import *  # noqa: F401, F403
from .verilator import *  # noqa: F401, F403
from .ovpsim import *  # noqa: F401, F403
from .vicuna import *  # noqa: F401, F403
from .benchmarks import *  # noqa: F401, F403
from .srecord import *  # noqa: F401, F403
from .layergen import *  # noqa: F401, F403
from .dtc import *  # noqa: F401, F403
from .corev import *  # noqa: F401, F403
from .cv32e40p import *  # noqa: F401, F403
