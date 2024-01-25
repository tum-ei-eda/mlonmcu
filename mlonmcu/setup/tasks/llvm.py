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

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_llvm(context: MlonMcuContext, params=None):
    if context.environment.has_toolchain("llvm"):
        return True
    if context.environment.has_framework("tvm"):
        user_vars = context.environment.vars
        use_tlcpack = user_vars.get("tvm.use_tlcpack", False)
        if not use_tlcpack:
            return True


@Tasks.provides(["llvm.install_dir"])
@Tasks.validate(_validate_llvm)
@Tasks.register(category=TaskType.MISC)
def install_llvm(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install LLVM."""
    llvmName = utils.makeDirName("llvm")
    llvmInstallDir = context.environment.paths["deps"].path / "install" / llvmName
    user_vars = context.environment.vars
    if "llvm.install_dir" in user_vars:  # TODO: also check command line flags?
        # TODO: WARNING
        llvmInstallDir = Path(user_vars["llvm.install_dir"])
    else:
        # TODO: share helper with riscv.py
        def _helper(url):
            # candidate_exts = [".zip", ".tar"]  # .tar.xz an .tar.gz also supported
            fullUrlSplit = url.split("/")
            llvmUrl = "/".join(fullUrlSplit[:-1])
            llvmFileName, llvmFileExtension = fullUrlSplit[-1].split(".", 1)
            return llvmUrl, llvmFileName, llvmFileExtension

        if "llvm.dl_url" in user_vars:
            llvmUrl, llvmFileName, llvmFileExtension = _helper(user_vars["llvm.dl_url"])
        else:
            llvmVersion = user_vars.get("llvm.version", "14.0.0")
            llvmDist = user_vars.get("llvm.distribution", "x86_64-linux-gnu-ubuntu-18.04")
            llvmUrl = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvmVersion}/"
            llvmFileName = f"clang+llvm-{llvmVersion}-{llvmDist}"
            llvmFileExtension = "tar.xz"
        llvmArchive = llvmFileName + "." + llvmFileExtension
        # if rebuild or not utils.is_populated(llvmInstallDir):
        # rebuild should only be triggered if the version/url changes but we can not detect that at the moment
        if not utils.is_populated(llvmInstallDir):
            utils.download_and_extract(llvmUrl, llvmArchive, llvmInstallDir, progress=verbose)
    context.cache["llvm.install_dir"] = llvmInstallDir
    context.export_paths.add(llvmInstallDir / "bin")
