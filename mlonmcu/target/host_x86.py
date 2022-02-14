"""MLonMCU Host/x86 Target definitions"""

import stat
from pathlib import Path

from .common import cli, execute
from .target import Target


class HostX86Target(Target):
    """Target using the x86 host system

    Mainly interesting to easy testing and debugging because benchmarking is not possible.
    """

    FEATURES = ["gdbserver"]

    DEFAULTS = {
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
    }

    def __init__(self, features=None, config=None, context=None):
        super().__init__("host_x86", features=features, config=config, context=context)
        self.gdb_path = "gdb"
        self.gdb_server_path = "gdbserver"

    @property
    def gdbserver_enable(self):
        return bool(self.config["gdbserver_enable"])

    @property
    def gdbserver_attach(self):
        return bool(self.config["gdbserver_attach"])

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

        return execute(program, *args, **kwargs)

    def get_arch(self):
        return "x86"


if __name__ == "__main__":
    cli(target=HostX86Target)
