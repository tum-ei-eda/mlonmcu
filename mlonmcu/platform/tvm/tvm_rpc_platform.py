from ...platform import Platform


class TvmRpcPlatform(Platform):
    """TVM RPC platform class."""

    FEATURES = Platform.FEATURES + [
        "tvm_rpc",
    ]

    DEFAULTS = {
        **Platform.DEFAULTS,
        "use_rpc": False,
        "rpc_key": None,
        "rpc_hostname": None,
        "rpc_port": None,
    }

    REQUIRED = Platform.REQUIRED + []

    @property
    def use_rpc(self):
        value = self.config["use_rpc"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def rpc_key(self):
        return self.config["rpc_key"]

    @property
    def rpc_hostname(self):
        return self.config["rpc_hostname"]

    @property
    def rpc_port(self):
        return self.config["rpc_port"]
