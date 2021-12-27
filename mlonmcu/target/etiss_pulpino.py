from .common import cli

SUPPORTED_FEATURES = ["attach", "trace"]

DEFAULT_CONFIG = {
    "etissvp.gdbserver_port": 2222,
    "etissvp.extra_args": "",
}

def lookupPrefix(cfg=None, env=None, context=None):
    prefix = None

    if cfg:
        if "riscv.dir" in cfg:
            pass

    if env:
        if "RISCV_DIR" in env:
            pass

    if context:
        if "MLONMCU_HOME" in env:
            pass

    if not prefix:
        prefix = ""

    return prefix

def lookupETISS(cfg={}, env=None):
    etiss = None

    if "etiss.dir":
        pass

class ETISSPulpinoTarget:

    def __init__(self, name, features=[], config={}, context=None):
        super().__init__("etiss_pulpino", features=features, config=config)
        # env = ?>??
        self.riscvPrefix = lookupPrefix(cfg=config, env=env, context=context)
        self.inspectProgram = "readelf"
        self.inspectprogramArgs = []

    def __repr__(self):
        return f"Target({self.name})"

    def exec(self, program, *args):
        pass
        # os.system(...)

    def inspect(self, program):
        pass
        # os.system(...)

if __main__ == "__main__":
    common.cli(target=ETISSPulpinoTarget)

## common

def cli(target):
    pass
    # Argparge
    # Env Vars
    # MLONMCU Context (can be disabled via --plain)
