"""MLonMCU ETISS/Pulpino Target definitions"""

import os
from pathlib import Path

from mlonmcu.context import MlonMcuContext

from .common import cli, execute
from .target import Target


def lookup_riscv_prefix(
    cfg: dict = None, env: os._Environ = None, context: MlonMcuContext = None
) -> Path:
    """Utility to find the directory where the RISCV GCC compiler toolchain is installed.

    Parameters
    ----------
    cfg : dict
        Optional config provided by the user.
    env : os._Environ
        Environment variables
    context : MlonMcuContext
        Optional context for looking up dependencies

    Returns
    -------
    path : pathlib.Path
        The path to the toolchain directory (if found).
    """
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
            with MlonMcuContext() as ctx:
                if ctx.cache:
                    if ctx.cache["riscv.dir"]:
                        prefix = ctx.cache["riscv.dir"]
        elif "RISCV_DIR" in env:
            prefix = env["RISCV_DIR"]

    if not prefix:
        prefix = ""

    return prefix


def lookup_etiss(
    cfg: dict = None, env: os._Environ = None, context: MlonMcuContext = None
) -> Path:
    """Utility to find the directory where the ETISS simulator is installed.

    Parameters
    ----------
    cfg : dict
        Optional config provided by the user.
    env : os._Environ
        Environment variables
    context : MlonMcuContext
        Optional context for looking up dependencies

    Returns
    -------
    path : pathlib.Path
        The path to the ETISS install directory (if found).
    """
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
            with MlonMcuContext() as ctx:
                if ctx.cache:
                    if ctx.cache["etiss.install_dir"]:
                        etiss = ctx.cache["etiss.install_dir"]
        if "ETISS_DIR" in env:
            etiss = env["ETISS_DIR"]

    if not etiss:
        etiss = ""

    return etiss


class ETISSPulpinoTarget(Target):
    """Target using a Pulpino-like VP running in the ETISS simulator"""

    FEATURES = ["gdbserver", "etissdbg", "trace"]

    DEFAULTS = {
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
        "debug_etiss": False,
        "trace_memory": False,
        "extra_args": "",
        "verbose": True,
    }
    REQUIRED = ["riscv_gcc.install_dir", "etiss.install_dir"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            "etiss_pulpino", features=features, config=config, context=context
        )
        # self.etiss_dir = lookup_etiss(cfg=config, env=self.env, context=self.context)
        # assert len(self.etiss_dir) > 0
        # self.riscv_prefix = lookup_riscv_prefix(
        #     cfg=config, env=self.env, context=self.context
        # )
        # assert len(self.riscv_prefix) > 0
        self.etiss_script = (
            Path(self.etiss_dir) / "examples" / "bare_etiss_processor" / "run_helper.sh"
        )
        # TODO: self.cmakeToolchainFile = ?
        # TODO: self.cmakeScript = ?

    @property
    def etiss_dir(self):
        return self.config["etiss.install_dir"]

    @property
    def riscv_prefix(self):
        return self.config["riscv_gcc.install_dir"]

    def exec(self, program, *args, **kwargs):
        """Use target to execute a executable with given arguments"""
        etiss_script_args = []
        config = DEFAULT_CONFIG
        for cfg, data in self.config.items():
            config[cfg] = data
        etiss_script_args.extend(
            config["etissvp.extra_args"].split(" ")
            if len(DEFAULT_CONFIG["etissvp.extra_args"]) > 0
            else []
        )

        # TODO: validate features (attach xor noattach!)
        if "etissdbg" in self.features:
            etiss_script_args.append("gdb")
        if "attach" in self.features:
            etiss_script_args.append("tgdb")
        if "noattach" in self.features:
            if "attach" not in self.features:
                etiss_script_args.append("tgdb")
            etiss_script_args.append("noattach")
        if "trace" in self.features:
            etiss_script_args.append("trace")
            etiss_script_args.append("nodmi")
        if bool(config["etissvp.verbose"]):
            etiss_script_args.append("v")

        return execute(
            self.etiss_script.resolve(), program, *etiss_script_args, *args, **kwargs
        )


if __name__ == "__main__":
    cli(target=ETISSPulpinoTarget)
