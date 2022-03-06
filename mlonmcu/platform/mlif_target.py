from mlonmcu.target.target import Target
from mlonmcu.target.host_x86 import HostX86Target
from mlonmcu.target.etiss_pulpino import EtissPulpinoTarget
from mlonmcu.target.corstone300 import Corstone300Target
from mlonmcu.target.spike import SpikeTarget
from mlonmcu.target.ovpsim import OVPSimTarget

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

        def get_platform_defs(self, platform):
            ret = super().get_platform_defs(platform)
            target_system = self.get_target_system()
            ret["TARGET_SYSTEM"] = target_system
            return ret

    return MlifTarget
