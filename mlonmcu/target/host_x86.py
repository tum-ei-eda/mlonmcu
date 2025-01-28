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
"""MLonMCU Host/x86 Target definitions"""

import stat
from pathlib import Path

from mlonmcu.config import str2bool
from mlonmcu.setup.utils import execute
from .common import cli
from .target import Target


class HostX86Target(Target):
    """Target using the x86 host system

    Mainly interesting to easy testing and debugging because benchmarking is not possible.
    """

    FEATURES = Target.FEATURES | {"gdbserver"}

    DEFAULTS = {
        **Target.DEFAULTS,
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
    }

    def __init__(self, name="host_x86", features=None, config=None):
        super().__init__(name, features=features, config=config)
        self.gdb_path = "gdb"
        self.gdb_server_path = "gdbserver"

    @property
    def gdbserver_enable(self):
        value = self.config["gdbserver_enable"]
        return str2bool(value)

    @property
    def gdbserver_attach(self):
        value = self.config["gdbserver_attach"]
        return str2bool(value)

    @property
    def gdbserver_port(self):
        return int(self.config["gdbserver_port"])

    def exec(self, program, *args, **kwargs):
        def make_executable(exe):
            f = Path(exe)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)

        make_executable(program)
        if self.gdbserver_enable:
            if self.gdbserver_attach:
                raise NotImplementedError
                # return execute(self.gdb_path, program, *args, **kwargs)
            else:
                comm = f"127.0.0.1:{self.gdbserver_port}"
                return execute(self.gdb_server_path, comm, program, *args, **kwargs)

        return execute(program, *args, **kwargs), []

    def get_arch(self):
        return "x86"


if __name__ == "__main__":
    cli(target=HostX86Target)
