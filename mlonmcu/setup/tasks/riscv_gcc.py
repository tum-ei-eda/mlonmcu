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

# import re
import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

# from .common import _validate_gcc

logger = get_logger()

Tasks = get_task_factory()


# def check_multilibs(riscvInstallDir, gccName, live=False, vext=False, pext=False):
#     gccExe = Path(riscvInstallDir) / "bin" / f"{gccName}-gcc"
#     out = utils.exec_getout(gccExe, "--print-multi-lib", print_output=False, live=live)
#     multilibs = []
#     if vext and pext:
#         default_multilib = "rv32gcpv/ilp32d" if "32" in gccName else "rv64gcpv/lp64d"  # TODO: improve this
#     elif vext:
#         default_multilib = "rv32gcv/ilp32d" if "32" in gccName else "rv64gcv/lp64d"  # TODO: improve this
#     elif pext:
#         default_multilib = "rv32gcp/ilp32d" if "32" in gccName else "rv64gcp/lp64d"  # TODO: improve this
#     else:
#         default_multilib = "rv32gc/ilp32d" if "32" in gccName else "rv64gc/lp64d"  # TODO: improve this
#     if len(out.split("\n")) < 3:
#         multilib = False
#     else:
#         multilib = True
#         libs = re.compile(r".+\/.+;@march=(.+)@mabi=(.+)").findall(out)
#         for arch, abi in libs:
#             multilibs.append(f"{arch}/{abi}")
#
#     return multilib, default_multilib, multilibs


def _validate_riscv_gcc(context: MlonMcuContext, params=None):
    if not (context.environment.has_toolchain("riscv_gcc")):
        return False
    return True


def _validate_riscv_gcc_vext(context: MlonMcuContext, params=None):
    if not (context.environment.has_toolchain("riscv_gcc_vext")):
        return False
    # if not context.environment.has_feature("vext"):
    #     return False
    return True


def _validate_riscv_gcc_pext(context: MlonMcuContext, params=None):
    if not (context.environment.has_toolchain("riscv_gcc_pext")):
        return False
    # if not context.environment.has_feature("pext"):
    #     return False
    return True


def _install_riscv_gcc(context, name, default_dl_url=None, default_version=None, verbose=False):
    user_vars = context.environment.vars
    flags = []
    riscvName = utils.makeDirName(name, flags=flags)
    riscvInstallDir = context.environment.deps_install_path / riscvName
    if f"{name}.install_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        riscvInstallDir = user_vars[f"{name}.install_dir"]
        # multilib = user_vars.get(f"{name}.multilib", None)
        # default_multilib = user_vars.get(f"{name}.default_multilib", None)
        # multilibs = user_vars.get(f"{name}.multilibs", None)
    else:

        def _helper(url):
            candidate_exts = [".zip", ".tar"]  # .tar.xz an .tar.gz also supported
            fullUrlSplit = url.split("/")
            riscvUrl = "/".join(fullUrlSplit[:-1])
            riscvFileName, riscvFileExtension = fullUrlSplit[-1].split(".", 1)
            riscvFileExtension = ""
            for ext in candidate_exts:
                if ext in riscvFileName:
                    riscvFileName, riscvFileExtension = riscvFileName.split(ext)
                    break
            return riscvUrl, riscvFileName, riscvFileExtension

        if f"{name}.dl_url" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars[f"{name}.dl_url"])
        elif default_dl_url:
            assert default_version is None
            riscvUrl, riscvFileName, riscvFileExtension = _helper(default_dl_url)
        else:
            riscvVersion = user_vars.get(f"{name}.version", "8.3.0-2020.04.0")
            riscvDist = user_vars.get(f"{name}.distribution", "x86_64-linux-ubuntu14")
            riscvUrl = "https://static.dev.sifive.com/dev-tools/"
            riscvFileName = f"riscv64-unknown-elf-gcc-{riscvVersion}-{riscvDist}"
            riscvFileExtension = "tar.gz"
        riscvArchive = riscvFileName + "." + riscvFileExtension
        # if rebuild or not utils.is_populated(riscvInstallDir):
        # rebuild should only be triggered if the version/url changes but we can not detect that at the moment
        if not utils.is_populated(riscvInstallDir):
            utils.download_and_extract(riscvUrl, riscvArchive, riscvInstallDir, progress=verbose)
    assert utils.is_populated(riscvInstallDir)
    if f"{name}.name" in user_vars:
        gccName = user_vars[f"{name}.name"]
    else:
        gccNames = ["riscv64-unknown-elf", "riscv32-unknown-elf"]
        gccName = None
        for name_ in gccNames:
            if (Path(riscvInstallDir) / name_).is_dir():
                gccName = name_
                break
    assert gccName is not None, "Toolchain name could not be determined automatically"
    # multilib_, default_multilib_, multilibs_ = check_multilibs(
    #     riscvInstallDir, gccName, live=verbose, vext=vext, pext=pext
    # )
    context.cache[f"{name}.install_dir", flags] = riscvInstallDir
    context.cache[f"{name}.name", flags] = gccName
    # context.cache["riscv_gcc.variant", flags] = variant
    # context.cache["riscv_gcc.multilib", flags] = multilib or multilib_
    # context.cache["riscv_gcc.default_multilib", flags] = default_multilib or default_multilib_
    # context.cache["riscv_gcc.multilibs", flags] = multilibs or multilibs_


@Tasks.provides(
    [
        "riscv_gcc.install_dir",
        "riscv_gcc.name",
    ]
)
@Tasks.validate(_validate_riscv_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install the RISCV GCC toolchain."""
    _install_riscv_gcc(context, "riscv_gcc", None, "8.3.0-2020.04.0", verbose=verbose)


@Tasks.provides(
    [
        "riscv_gcc_vext.install_dir",
        "riscv_gcc_vext.name",
    ]
)
@Tasks.validate(_validate_riscv_gcc_vext)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc_vext(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install the RISCV GCC toolchain with VEXT."""
    _install_riscv_gcc(
        context,
        "riscv_gcc_vext",
        "https://syncandshare.lrz.de/dl/fiGp4r3f6SZaC5QyDi6QUiNQ/rv32gcv_new.tar.gz",
        None,
        verbose=verbose,
    )


@Tasks.provides(
    [
        "riscv_gcc_pext.install_dir",
        "riscv_gcc_pext.name",
    ]
)
@Tasks.validate(_validate_riscv_gcc_pext)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc_pext(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install the RISCV GCC toolchain with VEXT."""
    _install_riscv_gcc(
        context,
        "riscv_gcc_pext",
        "https://syncandshare.lrz.de/dl/fiNvP4mzVQ8uDvgT9Yf2bqNk/rv32gcp.tar.xz",
        None,
        verbose=verbose,
    )
