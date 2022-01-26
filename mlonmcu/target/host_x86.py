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
        # TODO: self.cmakeToolchainFile = ?
        # TODO: self.cmakeScript = ?
        self.gdb_path = "gdb"
        self.gdb_server_path = "gdbserver"

    def exec(self, program, *args, **kwargs):
        if "attach" in self.features and "noattach" not in self.features:
            return execute(self.gdb_path, program, *args, **kwargs)
        if "noattach" in self.features:
            port = int(self.config["gdbserver_port"])
            comm = f"host:{port}"
            return execute(self.gdb_server_path, comm, program, *args, **kwargs)
        def make_executable(exe):
            f = Path(exe)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        make_executable(program)
        return execute(program, *args, **kwargs)


if __name__ == "__main__":
    cli(target=HostX86Target)
