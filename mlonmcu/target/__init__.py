"""MLonMCU target submodule"""

# from .target import Target
#
# SUPPORTED_TARGETS = {
#     "etiss/pulpino": Target("etiss/pulpino"),
#     "host": Target("host"),
# }  # TODO

from .etiss_pulpino import EtissPulpinoTarget
from .host_x86 import HostX86Target
from .corstone300 import Corstone300Target
from .spike import SpikeTarget
from .ovpsim import OVPSimTarget
from .esp32 import Esp32Target
from .esp32c3 import Esp32c3Target

SUPPORTED_TARGETS = {
    "etiss_pulpino": EtissPulpinoTarget,
    "host_x86": HostX86Target,
    "corstone300": Corstone300Target,
    "spike": SpikeTarget,
    "ovpsim": OVPSimTarget,
    "esp32": Esp32Target,  # TODO: in the long term we should just fetch the supported esp-idf boards at runtime
    "esp32c3": Esp32c3Target,  # TODO: in the long term we should just fetch the supported esp-idf boards at runtime
}
