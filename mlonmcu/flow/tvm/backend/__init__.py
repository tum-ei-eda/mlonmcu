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
from .backend import TVMBackend
from .tvmaot import TVMAOTBackend
from .tvmaotplus import TVMAOTPlusBackend
from .tvmcg import TVMCGBackend
from .tvmrt import TVMRTBackend
from .tvmllvm import TVMLLVMBackend

__all__ = ["TVMBackend", "TVMAOTBackend", "TVMAOTPlusBackend", "TVMCGBackend", "TVMRTBackend", "TVMLLVMBackend"]
