import os
from pathlib import Path
from .common import cli, execute
from .target import Target

SUPPORTED_FEATURES = ["attach", "noattach"]

DEFAULT_CONFIG = {
    "host.gdbserver_port": 2222,
}

class HostX86Target(Target):

    def __init__(self, features=[], config={}, context=None):
        super().__init__("host_x86", features=features, config=config, context=context)
        # TODO: self.cmakeToolchainFile = ?
        # TODO: self.cmakeScript = ?
        self.gdbPath = "gdb"
        self.gdbServerPath = "gdbserver"


    def exec(self, program, *args, **kwargs):
        print("exec", program, args, kwargs)
        config = DEFAULT_CONFIG
        for cfg in self.config:
            config[cfg] = self.config[cfg]
        if "attach" in self.features and "noattach" not in self.features:
            return execute(self.gdbPath, program, *args, **kwargs)
        elif "noattach" in self.features:
            port = int(config["host.gdbserver_port"])
            comm = f"host:{port}"
            return execute(self.gdbServerPath, comm, program, *args, **kwargs)
        return execute(program, *args, **kwargs)

if __name__ == "__main__":
    cli(target=HostX86Target)
