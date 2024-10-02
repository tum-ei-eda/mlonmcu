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
"""MLonMCU SSH Target definitions"""

import os
import re

# import tempfile
# import time
from pathlib import Path

import paramiko

from mlonmcu.config import str2bool
from .target import Target


class SSHTarget(Target):
    """TODO"""

    DEFAULTS = {
        **Target.DEFAULTS,
        "hostname": None,
        "port": 22,
        "username": None,
        "password": None,
        "ignore_known_hosts": True,
        "workdir": None,
    }

    @property
    def hostname(self):
        value = self.config["hostname"]
        assert value is not None, "hostname not defined"
        return value

    @property
    def port(self):
        value = self.config["port"]
        if isinstance(value, str):
            value = int(value)
        assert isinstance(value, int)
        return value

    @property
    def username(self):
        value = self.config["username"]
        return value

    @property
    def password(self):
        value = self.config["password"]
        return value

    @property
    def ignore_known_hosts(self):
        value = self.config["ignore_known_hosts"]
        return str2bool(value)

    @property
    def workdir(self):
        value = self.config["workdir"]
        if value is not None:
            if isinstance(value, str):
                value = Path(value)
            assert isinstance(value, Path)
        return value

    def __repr__(self):
        return f"SSHTarget({self.name})"

    # def check_remote(self):
    #     ssh = paramiko.SSHClient()
    #     try:
    #         ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password)
    #         # TODO: key_filename=key_file)
    #         return True
    #     except (BadHostKeyException, AuthenticationException, SSHException, socket.error) as e:
    #         print(e)  # TODO: remove
    #         return False
    #     raise NotImplementedError

    def create_remote_directory(self, ssh, path):
        command = f"mkdir -p {path}"
        stdin, stdout, stderr = ssh.exec_command(command)

    def copy_to_remote(self, ssh, src, dest):
        sftp = ssh.open_sftp()
        sftp.put(str(src), str(dest))
        sftp.close()

    def copy_from_remote(self, ssh, src, dest):
        sftp = ssh.open_sftp()
        sftp.get(str(src), str(dest))
        sftp.close()

    def parse_exit(self, out):
        exit_code = super().parse_exit(out)
        exit_match = re.search(r"SSH EXIT=(.*)", out)
        if exit_match:
            exit_code = int(exit_match.group(1))
        return exit_code

    def exec_via_ssh(self, program: Path, *args, cwd=os.getcwd(), **kwargs):
        # TODO: keep connection established!
        # self.check_remote()
        with paramiko.SSHClient() as ssh:
            # ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
            if self.ignore_known_hosts:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password)
            if self.workdir is None:
                raise NotImplementedError("temp workdir")
            else:
                self.create_remote_directory(ssh, self.workdir)
                workdir = self.workdir
            remote_program = workdir / program.name
            self.copy_to_remote(ssh, program, remote_program)
            args_str = " ".join(args)
            command = f"cd {workdir} && chmod +x {remote_program} && {remote_program} {args_str}; echo SSH EXIT=$?"
            stdin, stdout, stderr = ssh.exec_command(command)
            # print("stdin", stdin)
            # print("stdout", stdout)
            # print("stderr", stderr)
            output = stderr.read().strip() + stdout.read().strip()
            output = output.decode()
            if self.print_outputs:
                print("output", output)  # TODO: cleanup
        return output


# TODO: logger
