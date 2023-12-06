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

import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import _validate_gcc, get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_riscv_gcc(context: MlonMcuContext, params=None):
    if not _validate_gcc(context, params=params):
        return False
    if not (
        context.environment.has_target("etiss_pulpino")
        or context.environment.has_target("spike")
        or context.environment.has_target("ovpsim")
        or context.environment.has_target("ara")
    ):
        return False
    if params:
        vext = params.get("vext", False)
        pext = params.get("pext", False)
        user_vars = context.environment.vars
        multilib = user_vars.get("riscv_gcc.multilib", False)
        if vext and pext:
            return multilib
        elif vext:
            if not context.environment.has_feature("vext"):
                return False
        elif pext:
            if not context.environment.has_feature("pext"):
                return False
    return True


@Tasks.provides(["riscv_gcc.install_dir", "riscv_gcc.name", "riscv_gcc.variant"])
@Tasks.param("vext", [False, True])
@Tasks.param("pext", [False, True])
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
    multilib = user_vars.get("riscv_gcc.multilib", False)
    variant = user_vars.get("riscv_gcc.variant", "unknown")
    flags = utils.makeFlags((params["vext"], "vext"), (params["pext"], "pext"))
    # TODO: if the used gcc supports both pext and vext we do not need to download it 3 times!
    if multilib:
        riscvName = utils.makeDirName("riscv_gcc", flags=[])
    else:
        riscvName = utils.makeDirName("riscv_gcc", flags=flags)
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    if (not vext) and (not pext) and "riscv_gcc.install_dir_default" in user_vars:
        riscvInstallDir = user_vars["riscv_gcc.install_dir_default"]
    if vext and "riscv_gcc.install_dir_vext" in user_vars:
        assert not multilib, "Multilib toolchain does only support riscv_gcc.install_dir"
        riscvInstallDir = user_vars["riscv_gcc.install_dir_vext"]
    elif pext and "riscv_gcc.install_dir_pext" in user_vars:
        assert not multilib, "Multilib toolchain does only support riscv_gcc.install_dir"
        riscvInstallDir = user_vars["riscv_gcc.install_dir_pext"]
    elif "riscv_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        riscvInstallDir = user_vars["riscv_gcc.install_dir"]
    else:

        def _helper(url):
            fullUrlSplit = url.split("/")
            riscvUrl = "/".join(fullUrlSplit[:-1])
            riscvFileName, riscvFileExtension = fullUrlSplit[-1].split(".", 1)
            return riscvUrl, riscvFileName, riscvFileExtension

        if vext and "riscv_gcc.dl_url_vext" in user_vars:
            assert not multilib, "Multilib toolchain does only support riscv_gcc.dl_url"
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars["riscv_gcc.dl_url_vext"])
        elif pext and "riscv_gcc.dl_url_pext" in user_vars:
            assert not multilib, "Multilib toolchain does only support riscv_gcc.dl_url"
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars["riscv_gcc.dl_url_pext"])
        elif "riscv_gcc.dl_url" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars["riscv_gcc.dl_url"])
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
    assert utils.is_populated(riscvInstallDir)
    if "riscv_gcc.name" in user_vars:
        gccName = user_vars["riscv_gcc.name"]
    else:
        gccNames = ["riscv64-unknown-elf", "riscv32-unknown-elf"]
        gccName = None
        for name in gccNames:
            if (Path(riscvInstallDir) / name).is_dir():
                gccName = name
                break
    assert gccName is not None, "Toolchain name could not be determined automatically"
    context.cache["riscv_gcc.install_dir", flags] = riscvInstallDir
    context.cache["riscv_gcc.name", flags] = gccName
    context.cache["riscv_gcc.variant", flags] = variant
    context.export_paths.add(riscvInstallDir / "bin")
