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
"""IREE utils."""

from typing import Optional
from mlonmcu.logging import get_logger

logger = get_logger()


def parse_iree_version(iree_version: Optional[str] = None):
    if iree_version is None:
        logger.warning("iree.version undefined, assuming v3.3")
        major = 3
        minor = 3
    else:
        major, minor = map(int, iree_version.split(".", 1))
    return major, minor
