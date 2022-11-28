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
"""Definition of tasks used to dynamically install MLonMCU dependencies"""


from mlonmcu.setup.task import TaskFactory
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.logging import get_logger

logger = get_logger()

Tasks = TaskFactory()


def get_task_factory():
    return Tasks


def _validate_gcc(context: MlonMcuContext, params=None):
    return context.environment.has_toolchain("gcc")
