import os
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics

from mlonmcu.target.riscv.riscv import RISCVTarget

logger = get_logger()


class ABCTarget(RISCVTarget):

    FEATURES = RISCVTarget.FEATURES | set()

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
    }
    REQUIRED = RISCVTarget.REQUIRED | {"abc.exe", "abc.sw_dir"}

    def __init__(self, name="abc", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def abc_exe(self):
        return Path(self.config["abc.exe"])

    @property
    def sw_dir(self):
        return Path(self.config["abc.sw_dir"])

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        ret = execute(
            self.abc_exe.resolve(),
            program,
            *args,
            **kwargs,
        )
        return ret

    def parse_stdout(self, out, handle_exit=None):
        cycles = 42
        return cycles

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        total_cycles = self.parse_stdout(out, handle_exit=handle_exit)

        metrics = Metrics()
        metrics.add("Cycles", total_cycles)

        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["TOOLCHAIN_FILE"] = self.sw_dir / "toolchain.cmake"
        ret["TARGET_SYSTEM_FILE"] = self.sw_dir / "ABCTarget.cmake"

        return ret


if __name__ == "__main__":
    cli(target=ABCTarget)
