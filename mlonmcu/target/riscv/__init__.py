from .etiss_pulpino import EtissPulpinoTarget
from .etiss import EtissTarget, EtissRV32Target, EtissRV64Target
from .etiss_perf import EtissPerfTarget
from .spike import SpikeTarget, SpikeBMTarget, SpikePKTarget, SpikeRV32Target, SpikeRV32MinTarget, SpikeRV64Target
from .ovpsim import OVPSimTarget
from .corev_ovpsim import COREVOVPSimTarget
from .riscv_qemu import RiscvQemuTarget
from .gvsoc_pulp import GvsocPulpTarget
from .ara import AraTarget
from .ara_rtl import AraRtlTarget
from .cv32e40p import CV32E40PTarget
from .vicuna import VicunaTarget
from .canmv_k230_ssh import CanMvK230SSHTarget
from .tgc import TGCTarget

__all__ = [
    "EtissPulpinoTarget",
    "EtissTarget",
    "EtissPerfTarget",
    "EtissRV32Target",
    "EtissRV64Target",
    "SpikeTarget",
    "SpikeRV32Target",
    "SpikeRV32MinTarget",
    "SpikeRV64Target",
    "SpikeBMTarget",
    "SpikePKTarget",
    "OVPSimTarget",
    "COREVOVPSimTarget",
    "RiscvQemuTarget",
    "GvsocPulpTarget",
    "AraTarget",
    "AraRtlTarget",
    "CV32E40PTarget",
    "VicunaTarget",
    "CanMvK230SSHTarget",
    "TGCTarget",
]
