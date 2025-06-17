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
"""Definition of tasks used to dynamically install MLonMCU dependencies"""

import re
import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool

from .common import _validate_gcc, get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def check_multilibs(riscvInstallDir, gccName, live=False, vext=False, pext=False):
    gccExe = Path(riscvInstallDir) / "bin" / f"{gccName}-gcc"
    out = utils.execute(gccExe, "--print-multi-lib", live=live)
    multilibs = []
    if vext and pext:
        default_multilib = "rv32gcpv/ilp32d" if "32" in gccName else "rv64gcpv/lp64d"  # TODO: improve this
    elif vext:
        default_multilib = "rv32gcv/ilp32d" if "32" in gccName else "rv64gcv/lp64d"  # TODO: improve this
    elif pext:
        default_multilib = "rv32gcp/ilp32d" if "32" in gccName else "rv64gcp/lp64d"  # TODO: improve this
    else:
        default_multilib = "rv32gc/ilp32d" if "32" in gccName else "rv64gc/lp64d"  # TODO: improve this
    if len(out.split("\n")) < 3:
        multilib = False
    else:
        multilib = True
        libs = re.compile(r".+\/.+;@march=(.+)@mabi=(.+)").findall(out)
        for arch, abi in libs:
            multilibs.append(f"{arch}/{abi}")

    return multilib, default_multilib, multilibs


def _validate_riscv_gcc(context: MlonMcuContext, params=None):
    if not _validate_gcc(context, params=params):
        return False
    if not (
        context.environment.has_target("etiss_pulpino")
        or context.environment.has_target("etiss")
        or context.environment.has_target("spike")
        or context.environment.has_target("ovpsim")
        or context.environment.has_target("ara")
        or context.environment.has_target("tgc")
    ):
        return False
    if params:
        vext = params.get("vext", False)
        pext = params.get("pext", False)
        xlen = params.get("xlen", False)
        user_vars = context.environment.vars
        enable_rv32 = str2bool(user_vars.get("riscv_gcc.enable_rv32", True))
        enable_rv64 = str2bool(user_vars.get("riscv_gcc.enable_rv64", False))
        if vext:
            if not context.environment.has_feature("vext"):
                return False
        if pext:
            if not context.environment.has_feature("pext"):
                return False
        if xlen == 32 and not enable_rv32:
            return False
        if xlen == 64 and not enable_rv64:
            return False
    return True


@Tasks.provides(
    [
        "riscv_gcc_rv{xlen}.install_dir",
        "riscv_gcc_rv{xlen}.name",
        "riscv_gcc_rv{xlen}.variant",
        "riscv_gcc_rv{xlen}.multilib",
        "riscv_gcc_rv{xlen}.default_multilib",
        "riscv_gcc_rv{xlen}.multilibs",
    ]
)
@Tasks.param("vext", [False, True])
@Tasks.param("pext", [False, True])
@Tasks.param("xlen", [32, 64])  # TODO
@Tasks.validate(_validate_riscv_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install the RISCV GCC toolchain."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    vext = params["vext"]
    pext = params["pext"]
    xlen = params["xlen"]
    variant = user_vars.get(f"riscv_gcc_rv{xlen}.variant", "unknown")
    flags = utils.makeFlags((params["vext"], "vext"), (params["pext"], "pext"))
    # TODO: if the used gcc supports both pext and vext we do not need to download it 3 times!
    riscvName = utils.makeDirName(f"riscv_gcc_rv{xlen}", flags=flags)
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    if (not vext) and (not pext) and f"riscv_gcc_rv{xlen}.install_dir_default" in user_vars:
        riscvInstallDir = Path(user_vars[f"riscv_gcc_rv{xlen}.install_dir_default"])
        multilib = user_vars.get(f"riscv_gcc_rv{xlen}.multilib_default", None)
        default_multilib = user_vars.get(f"riscv_gcc_rv{xlen}.default_multilib_default", None)
        multilibs = user_vars.get(f"riscv_gcc_rv{xlen}.multilibs_default", None)
    elif vext and f"riscv_gcc_rv{xlen}.install_dir_vext" in user_vars:
        riscvInstallDir = Path(user_vars[f"riscv_gcc_rv{xlen}.install_dir_vext"])
        multilib = user_vars.get(f"riscv_gcc_rv{xlen}.multilib_vext", None)
        default_multilib = user_vars.get(f"riscv_gcc_rv{xlen}.default_multilib_vext", None)
        multilibs = user_vars.get(f"riscv_gcc_rv{xlen}.multilibs_vext", None)
    elif pext and f"riscv_gcc_rv{xlen}.install_dir_pext" in user_vars:
        riscvInstallDir = Path(user_vars[f"riscv_gcc_rv{xlen}.install_dir_pext"])
        multilib = user_vars.get(f"riscv_gcc_rv{xlen}.multilib_pext", None)
        default_multilib = user_vars.get(f"riscv_gcc_rv{xlen}.default_multilib_pext", None)
        multilibs = user_vars.get(f"riscv_gcc_rv{xlen}.multilibs_pext", None)
    elif f"riscv_gcc_rv{xlen}.install_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        riscvInstallDir = Path(user_vars[f"riscv_gcc_rv{xlen}.install_dir"])
        multilib = user_vars.get(f"riscv_gcc_rv{xlen}.multilib", None)
        default_multilib = user_vars.get(f"riscv_gcc_rv{xlen}.default_multilib", None)
        multilibs = user_vars.get(f"riscv_gcc_rv{xlen}.multilibs", None)
    else:

        def _helper(url):
            fullUrlSplit = url.split("/")
            riscvUrl = "/".join(fullUrlSplit[:-1])
            riscvFileName, riscvFileExtension = fullUrlSplit[-1].split(".", 1)
            return riscvUrl, riscvFileName, riscvFileExtension

        if vext and f"riscv_gcc_rv{xlen}.dl_url_vext" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars[f"riscv_gcc_rv{xlen}.dl_url_vext"])
        elif pext and f"riscv_gcc_rv{xlen}.dl_url_pext" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars[f"riscv_gcc_rv{xlen}.dl_url_pext"])
        elif f"riscv_gcc_rv{xlen}.dl_url" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars[f"riscv_gcc_rv{xlen}.dl_url"])
        else:
            riscvVersion = (
                user_vars["riscv.version"]
                if "riscv.version" in user_vars
                else ("8.3.0-2020.04.0" if not vext else "10.2.0-2020.12.8")
            )
            riscvDist = (
                user_vars["riscv.distribution"] if "riscv.distribution" in user_vars else "x86_64-linux-ubuntu14"
            )
            if vext:
                subdir = "v" + ".".join(riscvVersion.split("-")[1].split(".")[:-1])
                riscvUrl = "https://static.dev.sifive.com/dev-tools/freedom-tools/" + subdir + "/"
                riscvFileName = f"riscv64-unknown-elf-toolchain-{riscvVersion}-{riscvDist}"
            else:
                riscvUrl = "https://static.dev.sifive.com/dev-tools/"
                riscvFileName = f"riscv64-unknown-elf-gcc-{riscvVersion}-{riscvDist}"
            riscvFileExtension = "tar.gz"
        riscvArchive = riscvFileName + "." + riscvFileExtension
        # if rebuild or not utils.is_populated(riscvInstallDir):
        # rebuild should only be triggered if the version/url changes but we can not detect that at the moment
        if not utils.is_populated(riscvInstallDir):
            utils.download_and_extract(riscvUrl, riscvArchive, riscvInstallDir, progress=verbose)
        # workaround for gnu subdir ins tc downloads
        gnu_dir = riscvInstallDir / "gnu"
        if gnu_dir.is_dir():
            import shutil

            shutil.move(riscvInstallDir, riscvInstallDir.parent / f"{riscvInstallDir.name}.old")
            shutil.move(riscvInstallDir.parent / f"{riscvInstallDir.name}.old" / "gnu", riscvInstallDir)

        multilib = user_vars.get(f"riscv_gcc_rv{xlen}.multilib", None)
        default_multilib = user_vars.get(f"riscv_gcc_rv{xlen}.default_multilib", None)
        multilibs = user_vars.get(f"riscv_gcc_rv{xlen}.multilibs", None)
    assert utils.is_populated(riscvInstallDir)
    if f"riscv_gcc_rv{xlen}.name" in user_vars:
        gccName = user_vars[f"riscv_gcc_rv{xlen}.name"]
    else:
        gccNames = ["riscv64-unknown-elf", "riscv32-unknown-elf", "riscv64-unknown-linux-musl"]
        gccName = None
        for name in gccNames:
            if (Path(riscvInstallDir) / name).is_dir():
                gccName = name
                break
    assert gccName is not None, "Toolchain name could not be determined automatically"
    multilib_, default_multilib_, multilibs_ = check_multilibs(
        riscvInstallDir, gccName, live=verbose, vext=vext, pext=pext
    )
    context.cache[f"riscv_gcc_rv{xlen}.install_dir", flags] = riscvInstallDir
    context.cache[f"riscv_gcc_rv{xlen}.name", flags] = gccName
    context.cache[f"riscv_gcc_rv{xlen}.variant", flags] = variant
    context.cache[f"riscv_gcc_rv{xlen}.multilib", flags] = multilib or multilib_
    context.cache[f"riscv_gcc_rv{xlen}.default_multilib", flags] = default_multilib or default_multilib_
    context.cache[f"riscv_gcc_rv{xlen}.multilibs", flags] = multilibs or multilibs_
    context.export_paths.add(riscvInstallDir / "bin")
