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
"""TVM RPC Platform"""
from mlonmcu.config import str2bool
from ...platform import Platform


class TvmRpcPlatform(Platform):
    """TVM RPC platform class."""

    FEATURES = Platform.FEATURES | {
        "tvm_rpc",
    }

    DEFAULTS = {
        **Platform.DEFAULTS,
        "use_rpc": False,
        "rpc_key": None,
        "rpc_hostname": None,
        "rpc_port": None,
    }

    @property
    def use_rpc(self):
        value = self.config["use_rpc"]
        return str2bool(value)

    @property
    def rpc_key(self):
        return self.config["rpc_key"]

    @property
    def rpc_hostname(self):
        return self.config["rpc_hostname"]

    @property
    def rpc_port(self):
        return self.config["rpc_port"]
