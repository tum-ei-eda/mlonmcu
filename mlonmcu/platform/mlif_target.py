import os
from enum import IntEnum

from mlonmcu.target.target import Target
from mlonmcu.target.host_x86 import HostX86Target
from mlonmcu.target.etiss_pulpino import EtissPulpinoTarget
from mlonmcu.target.corstone300 import Corstone300Target
from mlonmcu.target.spike import SpikeTarget
from mlonmcu.target.ovpsim import OVPSimTarget
from mlonmcu.logging import get_logger

logger = get_logger()

MLIF_TARGET_REGISTRY = {}


def register_mlif_target(target_name, t, override=False):
    global MLIF_TARGET_REGISTRY

    if target_name in MLIF_TARGET_REGISTRY and not override:
        raise RuntimeError(f"MLIF target {target_name} is already registered")
    MLIF_TARGET_REGISTRY[target_name] = t


def get_mlif_targets():
    return MLIF_TARGET_REGISTRY


register_mlif_target("host_x86", HostX86Target)
register_mlif_target("etiss_pulpino", EtissPulpinoTarget)
register_mlif_target("corstone300", Corstone300Target)
register_mlif_target("spike", SpikeTarget)
register_mlif_target("ovpsim", OVPSimTarget)

class MlifExitCode(IntEnum):
    ERROR = 0x10
    INVALID_SIZE = 0x11
    OUTPUT_MISSMATCH = 0x12

    @classmethod
    def values(cls):
        return list(map(int, cls))

def create_mlif_target(name, platform, base=Target):
    class MlifTarget(base):  # This is not ideal as we will have multiple different MlifTarget classes

        FEATURES = base.FEATURES + []

        DEFAULTS = {
            **base.DEFAULTS,
        }
        REQUIRED = base.REQUIRED + []

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.validation_result = None


        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            # This is wrapper around the original exec function to catch special return codes thrown by the inout data feature
            # TODO: catch edge cases: no input data available (skipped) and no return code (real hardware)
            if self.platform.validate_outputs:
                def handle_exit(code):
                    if code == 0:
                        self.validation_result = True
                    else:
                        if code in MlifExitCode.values():
                            reason = MlifExitCode(code).name
                            logger.error("A platform error occured during the simulation. Reason: %s", reason)
                            self.validation_result = False
                            if not self.platform.fail_on_error:
                                code = 0
                    return code
                kwargs["handle_exit"] = handle_exit
            return super().exec(program, *args, cwd=cwd, **kwargs)

        def get_metrics(self, elf, directory, verbose=False):
            metrics = super().get_metrics(elf, directory, verbose=verbose)
            if self.platform.validate_outputs:
                metrics.add("Validation", self.validation_result)
            return metrics

        def get_platform_defs(self, platform):
            ret = super().get_platform_defs(platform)
            target_system = self.get_target_system()
            ret["TARGET_SYSTEM"] = target_system
            return ret


    return MlifTarget
