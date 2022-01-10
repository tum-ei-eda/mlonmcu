import os
from pathlib import Path
from .common import cli, execute
from .target import Target

SUPPORTED_FEATURES = ["etissdbg", "attach", "noattach", "trace"]

DEFAULT_CONFIG = {
    "etissvp.gdbserver_port": 2222,  # TODO: make this configurable in ETISS
    "etissvp.extra_args": "",
    "etissvp.verbose": True,
}

def lookupRISCVPrefix(cfg=None, env=None, context=None):
    prefix = None

    if cfg:
        if "riscv.dir" in cfg:
            prefix = cfg["riscv.dir"]

    if context:
        if context.cache:
            if context.cache["riscv.dir"]:
                prefix = context.cache["riscv.dir"]

    if env:
        if "MLONMCU_HOME" in env:
            with mlonmcu.context.MlonMcuContext() as context:
                if context.cache:
                    if context.cache["riscv.dir"]:
                        prefix = context.cache["riscv.dir"]
        elif "RISCV_DIR" in env:
            prefix = env["RISCV_DIR"]

    if not prefix:
        prefix = ""

    return prefix

def lookupETISS(cfg={}, env=None, context=None):
    etiss = None

    if cfg:
        if "etiss.dir" in cfg:
            etiss = cfg["etiss.dir"]

    if context:  # TODO: feature flags?
        if context.cache:
            if context.cache["etiss.install_dir"]:
                etiss = context.cache["etiss.install_dir"]
    if env:
        if "MLONMCU_HOME" in env:
            with mlonmcu.context.MlonMcuContext() as context:
                if context.cache:
                    if context.cache["etiss.install_dir"]:
                        etiss = context.cache["etiss.install_dir"]
        if "ETISS_DIR" in env:
            etiss = env["ETISS_DIR"]

    if not etiss:
        etiss = ""

    return etiss


class ETISSPulpinoTarget(Target):

    def __init__(self, features=[], config={}, context=None):
        super().__init__("etiss_pulpino", features=features, config=config, context=context)
        self.etissDir = lookupETISS(cfg=config, env=self.env, context=self.context)
        assert len(self.etissDir) > 0
        self.riscvPrefix = lookupRISCVPrefix(cfg=config, env=self.env, context=self.context)
        assert len(self.riscvPrefix) > 0
        self.etissScript = Path(self.etissDir) / "examples" / "bare_etiss_processor" / "run_helper.sh"
        # TODO: self.cmakeToolchainFile = ?
        # TODO: self.cmakeScript = ?

    def exec(self, program, *args, **kwargs):
        print("exec", program, args, kwargs)
        etissScriptArgs = []
        config = DEFAULT_CONFIG
        for cfg in self.config:
            config[cfg] = self.config[cfg]
        etissScriptArgs.extend(config["etissvp.extra_args"].split(" ") if len(DEFAULT_CONFIG["etissvp.extra_args"]) > 0 else [])

        # TODO: validate features (attach xor noattach!)
        if "etissdbg" in self.features:
            etissScriptArgs.append("gdb")
        if "attach" in self.features:
            etissScriptArgs.append("tgdb")
        if "noattach" in self.features:
            if not "attach" in self.features:
                etissScriptArgs.append("tgdb")
            etissScriptArgs.append("noattach")
        if "trace" in self.features:
            etissScriptArgs.append("trace")
            etissScriptArgs.append("nodmi")
        if bool(config["etissvp.verbose"]):
            etissScriptArgs.append("v")

        return execute(self.etissScript.name, program, *etissScriptArgs, *args, **kwargs)

if __name__ == "__main__":
    cli(target=ETISSPulpinoTarget)
