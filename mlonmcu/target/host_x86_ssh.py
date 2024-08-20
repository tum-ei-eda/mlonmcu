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

from .common import cli
from .ssh_target import SSHTarget
from .host_x86 import HostX86Target


class HostX86SSHTarget(SSHTarget, HostX86Target):
    """TODO"""

    FEATURES = SSHTarget.FEATURES | HostX86Target.FEATURES  # TODO: do not allow gdbserver

    DEFAULTS = {
        **SSHTarget.DEFAULTS,
        **HostX86Target.DEFAULTS,
    }

    def __init__(self, name="host_x86_ssh", features=None, config=None):
        super().__init__(name, features=features, config=config)

    def exec(self, program, *args, handle_exit=None, **kwargs):
        if self.gdbserver_enable:
            raise NotImplementedError("gdbserver via ssh")

        output = self.exec_via_ssh(program, *args, **kwargs)
        if handle_exit:
            exit_code = handle_exit(0, out=output)
            assert exit_code == 0
        return output, []

    def get_target_system(self):
        return "host_x86"


if __name__ == "__main__":
    cli(target=HostX86SSHTarget)
