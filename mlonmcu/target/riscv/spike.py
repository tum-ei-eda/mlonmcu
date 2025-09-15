#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
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
"""MLonMCU Spike Target definitions"""

import os
import re
import time
import tempfile
import multiprocessing
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.setup import utils
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from mlonmcu.config import pick_first, str2bool
from .riscv_pext_target import RVPTarget
from .riscv_vext_target import RVVTarget
from .riscv_bext_target import RVBTarget
from .util import update_extensions, sort_extensions_canonical, join_extensions

logger = get_logger()


def _build_spike_pk(
    dest, pk_src, riscv_gcc_prefix, riscv_gcc_name, arch, abi, verbose=False, threads=multiprocessing.cpu_count()
):
    with tempfile.TemporaryDirectory() as temp_dir:
        build_dir = Path(temp_dir)
        args = []
        args.append(f"--with-arch={arch}")
        args.append(f"--with-abi={abi}")
        args.append("--prefix=" + str(riscv_gcc_prefix))
        args.append("--host=" + str(riscv_gcc_name))
        env = os.environ.copy()
        env["PATH"] = str(Path(riscv_gcc_prefix) / "bin") + ":" + env["PATH"]
        utils.execute(
            str(pk_src / "configure"),
            *args,
            cwd=build_dir,
            env=env,
            live=False,
        )
        utils.make(cwd=build_dir, threads=threads, live=verbose, env=env)
        utils.copy(build_dir / "pk", dest)


def filter_unsupported_extensions(exts, legacy: bool = False):
    assert isinstance(exts, set)
    REPLACEMENTS = {
        r"zpsfoperand": "p",
        r"zpn": "p",
        r"zbpo": "p",
        # r"p": ["p", "b"],
        # r"p": ["p", "zba", "zbb", "zbc", "zbs"],
    }
    if legacy:
        REPLACEMENTS.update(
            {
                r"zve\d\d[xfd]": "v",
                r"zvl\d+b": None,
            }
        )
    ret = set()
    for ext in exts:
        ignore = False
        for key, value in REPLACEMENTS.items():
            m = re.compile(key).match(ext)
            if m:
                if value:
                    if isinstance(value, list):
                        assert len(value) > 0
                        ret |= set(value)
                    else:
                        ret.add(value)
                ignore = True
        if not ignore:
            ret.add(ext)
    ret = set(ret)

    return ret


class SpikeBaseTarget(RVPTarget, RVVTarget, RVBTarget):
    """Target using the riscv-isa-sim (Spike) RISC-V simulator."""

    FEATURES = RVPTarget.FEATURES | RVVTarget.FEATURES | RVBTarget.FEATURES | {"cachesim", "log_instrs"}

    DEFAULTS = {
        **RVPTarget.DEFAULTS,
        **RVVTarget.DEFAULTS,
        **RVBTarget.DEFAULTS,
        "legacy": True,
    }
    REQUIRED = RVPTarget.REQUIRED | RVVTarget.REQUIRED | RVBTarget.REQUIRED

    OPTIONAL = RVPTarget.OPTIONAL | RVVTarget.OPTIONAL | RVBTarget.OPTIONAL | {"spike.exe"}

    def __init__(self, name, features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def legacy(self):
        value = self.config["legacy"]
        return str2bool(value)

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def extensions(self):
        # exts = RVPTarget.extensions(self) + RVVTarget.extensions(self)
        exts = super().extensions
        return update_extensions(
            exts,
        )

    @property
    def isa(self):
        exts = self.extensions
        if not self.legacy:
            exts.add("zicntr")
        exts = filter_unsupported_extensions(exts, legacy=self.legacy)
        exts_str = join_extensions(sort_extensions_canonical(exts, lower=True))
        return f"rv{self.xlen}{exts_str}"

    def get_spike_args(self):
        spike_args = []
        spike_args.append(f"--isa={self.isa}")

        if self.enable_vext:
            assert self.vlen < 8192, "Spike does not support VLEN >= 8192"
            if self.legacy:
                spike_args.append(f"--varch=vlen:{self.vlen},elen:{self.elen}")
        else:
            # assert self.vlen == 0
            pass

        return spike_args

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        raise NotImplementedError()  # TODO: abstract!

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        sim_insns = re.search(r"(\d*) cycles", out)
        if sim_insns:  # PK only
            sim_insns = int(float(sim_insns.group(1)))
            metrics.add("Simulated Instructions", sim_insns, True)

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            # TODO: before or after?
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        start_time = time.time()
        if self.print_outputs:
            out, artifacts = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out, artifacts = self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        # size instead of readelf?

        # TODO: get exit code
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)

        if metrics.has("Simulated Instructions"):
            sim_insns = metrics.get("Simulated Instructions")
            if diff > 0:
                metrics.add("MIPS", (sim_insns / diff) / 1e6, True)

        return metrics, out, artifacts

    def get_platform_defs(self, platform):
        ret = {}
        ret.update(RVPTarget.get_platform_defs(self, platform))
        ret.update(RVVTarget.get_platform_defs(self, platform))
        ret.update(RVBTarget.get_platform_defs(self, platform))
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        if backend in SUPPORTED_TVM_BACKENDS:
            if optimized_layouts:
                if self.enable_pext or self.enable_vext:
                    ret.update(
                        {
                            # Warning: passing kernel layouts does not work with upstream TVM
                            # TODO: allow passing map?
                            "desired_layout": "NHWC:HWOI",
                        }
                    )
        return ret


class SpikePKTarget(SpikeBaseTarget):

    DEFAULTS = {
        **SpikeBaseTarget.DEFAULTS,
        "spikepk_extra_args": [],
        "build_pk": False,
    }

    OPTIONAL = SpikeBaseTarget.OPTIONAL | {
        "spike.pk",
        "spike.pk_rv32",
        "spike.pk_rv64",
        "spikepk.src_dir",
        "spike_pk.pk",
        "spike_pk.pk_rv32",
        "spike_pk.pk_rv64",
    }

    @property
    def build_pk(self):
        value = self.config["build_pk"]
        return str2bool(value)

    @property
    def spike_pk(self):
        print("spike_pk")
        ret = Path(
            pick_first(
                self.config,
                [
                    f"{self.name}.pk_rv{self.xlen}",
                    f"{self.name}.pk",
                    f"spike.pk_rv{self.xlen}",
                    "spike.pk",
                ],
            )
        )
        print("ret", ret)
        # input(">")
        return ret

    @property
    def spike_pk_src_dir(self):
        value = self.config["spikepk.src_dir"]
        return value if value is None else Path(value)

    @property
    def spikepk_extra_args(self):
        return self.config["spikepk_extra_args"]

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        spike_args = self.get_spike_args()
        spikepk_args = []

        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            spike_args.extend(extra_args)

        spikepk_args.append("-s")

        if len(self.spikepk_extra_args) > 0:
            if isinstance(self.spikepk_extra_args, str):
                extra_args = self.spikepk_extra_args.split(" ")
            else:
                extra_args = self.spikepk_extra_args
            spikepk_args.extend(extra_args)  # I rename args to extra_args because otherwise it overwrites *args

        if self.timeout_sec > 0:
            raise NotImplementedError

        if self.build_pk:
            # TODO: tempdir
            assert cwd is not None
            cwd = Path(cwd)
            assert cwd.is_dir()
            pk = cwd / ".temp_spike_pk"
            arch = self.isa
            if "zicsr" not in arch:
                arch += "_zicsr"
            if "zifencei" not in arch:
                arch += "_zifencei"
            _build_spike_pk(pk, self.spike_pk_src_dir, self.riscv_gcc_prefix, self.riscv_gcc_basename, arch, self.abi)
        else:
            pk = self.spike_pk.resolve()

        ret = execute(
            self.spike_exe.resolve(),
            *spike_args,
            pk,
            *spikepk_args,
            program,
            *args,
            cwd=cwd,
            **kwargs,
        )
        return ret, []


class SpikeBMTarget(SpikeBaseTarget):

    DEFAULTS = {
        **SpikeBaseTarget.DEFAULTS,
        "htif": True,
        "htif_nano": True,
        "htif_wrap": True,
        "htif_argv": False,
    }

    # def get_target_system(self):
    #     return "generic_riscv_bm"

    def __init__(self, name="spike_bm", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def htif(self):
        value = self.config["htif"]
        return str2bool(value)

    @property
    def htif_nano(self):
        value = self.config["htif_nano"]
        return str2bool(value)

    @property
    def htif_wrap(self):
        value = self.config["htif_wrap"]
        return str2bool(value)

    @property
    def htif_argv(self):
        value = self.config["htif_argv"]
        return str2bool(value)

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        spike_args = self.get_spike_args()

        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            spike_args.extend(extra_args)

        if self.timeout_sec > 0:
            raise NotImplementedError

        ret = execute(
            self.spike_exe.resolve(),
            *spike_args,
            program,
            *args,
            cwd=cwd,
            **kwargs,
        )
        return ret, []

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret.update(
            {
                "HTIF": self.htif,
                "HTIF_NANO": self.htif_nano,
                "HTIF_WRAP": self.htif_wrap,
                "HTIF_ARGV": self.htif_argv,
            }
        )
        return ret


class SpikeTarget(SpikePKTarget):  # Alias for compatibility reasons
    def __init__(self, name="spike", features=None, config=None):
        super().__init__(name, features=features, config=config)


class SpikeRV32Target(SpikeTarget):
    """32-bit version of spike target"""

    DEFAULTS = {
        **SpikeTarget.DEFAULTS,
        "xlen": 32,
        "vlen": 0,  # vectorization=off
        # "elen": 32,
    }

    def __init__(self, name="spike_rv32", features=None, config=None):
        super().__init__(name, features=features, config=config)


class SpikeRV32MinTarget(SpikeRV32Target):
    """32-bit integer-only version of spike target"""

    DEFAULTS = {
        **SpikeTarget.DEFAULTS,
        "xlen": 32,
        "vlen": 0,  # vectorization=off
        "fpu": "none",
        "compressed": False,
        "atomic": False,
        "embedded_vext": True,
        "build_pk": True,
        # "elen": 32,
    }

    OPTIONAL = SpikeRV32Target.OPTIONAL | {
        "riscv_gcc_rv32_min.install_dir",
        "riscv_gcc_rv32im.install_dir",
        "riscv_gcc_rv32im_ilp32.install_dir",
        "riscv_gcc_rv32im_zve64x.install_dir",
        "riscv_gcc_rv32im_zve64x_ilp32.install_dir",
    }

    def __init__(self, name="spike_rv32_min", features=None, config=None):
        super().__init__(name, features=features, config=config)


class SpikeRV64Target(SpikeTarget):
    """64-bit version of spike target"""

    DEFAULTS = {
        **SpikeTarget.DEFAULTS,
        "xlen": 64,
        "vlen": 0,  # vectorization=off
        # "elen": 64,
    }

    def __init__(self, name="spike_rv64", features=None, config=None):
        super().__init__(name, features=features, config=config)


if __name__ == "__main__":
    cli(target=SpikeTarget)
