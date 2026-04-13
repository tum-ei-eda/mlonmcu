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
"""Definition of tasks used to dynamically install onnx2c dependencies."""

import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils

from .common import get_task_factory

Tasks = get_task_factory()


def _validate_onnx2c(context: MlonMcuContext, params=None):
    del params
    if not context.environment.has_backend("onnx2c"):
        return False
    return True


@Tasks.provides(["onnx2c.src_dir"])
@Tasks.validate(_validate_onnx2c)
@Tasks.register(category=TaskType.BACKEND)
def clone_onnx2c(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the onnx2c repository."""
    del params, verbose, threads
    onnx2c_name = utils.makeDirName("onnx2c")
    onnx2c_src_dir = context.environment.paths["deps"].path / "src" / onnx2c_name
    if rebuild or not utils.is_populated(onnx2c_src_dir):
        if "onnx2c" not in context.environment.repos:
            raise RuntimeError(
                "Missing repository definition for 'onnx2c' in environment.yml (repos section)."
            )
        onnx2c_repo = context.environment.repos["onnx2c"]
        utils.clone_wrapper(onnx2c_repo, onnx2c_src_dir, refresh=rebuild)
    context.cache["onnx2c.src_dir"] = onnx2c_src_dir


@Tasks.needs(["onnx2c.src_dir"])
@Tasks.optional(["cmake.exe"])
@Tasks.provides(["onnx2c.build_dir", "onnx2c.exe"])
@Tasks.validate(_validate_onnx2c)
@Tasks.register(category=TaskType.BACKEND)
def build_onnx2c(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Build the onnx2c compiler executable."""
    del params
    user_vars = context.environment.vars
    onnx2c_name = utils.makeDirName("onnx2c")
    onnx2c_src_dir = None
    if "onnx2c.src_dir" in user_vars:
        onnx2c_src_dir = Path(user_vars["onnx2c.src_dir"])
    else:
        try:
            onnx2c_src_dir = context.lookup("onnx2c.src_dir", ())
        except KeyError:
            src_base_dir = context.environment.paths["deps"].path / "src"
            candidate = src_base_dir / onnx2c_name
            if utils.is_populated(candidate):
                onnx2c_src_dir = candidate
    if onnx2c_src_dir is None:
        raise RuntimeError(
            "Unable to resolve onnx2c source directory. "
            "Run 'mlonmcu setup --task clone_onnx2c --rebuild' first, or set 'onnx2c.src_dir' in vars."
        )

    onnx2c_build_dir = context.environment.paths["deps"].path / "build" / onnx2c_name
    onnx2c_install_dir = context.environment.paths["deps"].path / "install" / onnx2c_name
    onnx2c_exe = onnx2c_install_dir / "onnx2c"
    cmake_exe = context.cache.get("cmake.exe")

    if rebuild or not utils.is_populated(onnx2c_build_dir) or not onnx2c_exe.is_file():
        utils.mkdirs(onnx2c_build_dir)
        utils.cmake(
            "-DCMAKE_BUILD_TYPE=Release",
            str(onnx2c_src_dir),
            cwd=onnx2c_build_dir,
            live=verbose,
            cmake_exe=cmake_exe,
        )
        utils.make(cwd=onnx2c_build_dir, threads=threads, live=verbose)
        utils.mkdirs(onnx2c_install_dir)
        utils.copy(onnx2c_build_dir / "onnx2c", onnx2c_exe)

    context.cache["onnx2c.build_dir"] = onnx2c_build_dir
    context.cache["onnx2c.exe"] = onnx2c_exe
    context.export_paths.add(onnx2c_exe.parent)
