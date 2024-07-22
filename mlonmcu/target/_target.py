# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from .riscv import (
    EtissPulpinoTarget,
    SpikeTarget,
    OVPSimTarget,
    COREVOVPSimTarget,
    RiscvQemuTarget,
    GvsocPulpTarget,
    EtissTarget,
    AraTarget,
    AraRtlTarget,
    CV32E40PTarget,
    VicunaTarget,
    CanMvK230SSHTarget,
)
from .arm import Corstone300Target
from .host_x86 import HostX86Target
from .host_x86_ssh import HostX86SSHTarget

TARGET_REGISTRY = {}


def register_target(target_name, t, override=False):
    global TARGET_REGISTRY

    if target_name in TARGET_REGISTRY and not override:
        raise RuntimeError(f"Target {target_name} is already registered")
    TARGET_REGISTRY[target_name] = t


def get_targets():
    return TARGET_REGISTRY


register_target("etiss_pulpino", EtissPulpinoTarget)
register_target("etiss", EtissTarget)
register_target("host_x86", HostX86Target)
register_target("host_x86_ssh", HostX86SSHTarget)
register_target("corstone300", Corstone300Target)
register_target("spike", SpikeTarget)
register_target("ovpsim", OVPSimTarget)
register_target("corev_ovpsim", COREVOVPSimTarget)
register_target("riscv_qemu", RiscvQemuTarget)
register_target("gvsoc_pulp", GvsocPulpTarget)
register_target("ara", AraTarget)
register_target("ara_rtl", AraRtlTarget)
register_target("cv32e40p", CV32E40PTarget)
register_target("vicuna", VicunaTarget)
register_target("canmv_k230_ssh", CanMvK230SSHTarget)
