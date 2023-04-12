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

from mlonmcu.target.target import Target

from mlonmcu.logging import get_logger
from mlonmcu.utils import filter_none

logger = get_logger()


class TemplateMicroTvmPlatformTarget(Target):
    REQUIRED = Target.REQUIRED | {"tvm.build_dir"}

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = None
        self.option_names = []

    def get_project_options(self):
        def _bool_helper(x):
            return str(x).lower() if isinstance(x, bool) else x

        return filter_none(
            {
                key: _bool_helper(getattr(self, key) if hasattr(self, key) else self.config.get(key, None))
                for key in self.option_names
            }
        )

    def update_environment(self, env):
        pass
