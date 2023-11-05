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

import os
import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_cv32e40p(context: MlonMcuContext, params=None):
    return context.environment.has_target("cv32e40p")


@Tasks.provides(["cv32e40p.src_dir"])
@Tasks.validate(_validate_cv32e40p)
@Tasks.register(category=TaskType.TARGET)
def clone_cv32e40p(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the cv32e40p rtl."""
    name = utils.makeDirName("cv32e40p")
    srcDir = context.environment.paths["deps"].path / "src" / name
    user_vars = context.environment.vars
    # if "cv32e40p.verilator_executable" in user_vars:  # TODO: also check command line flags?
    if "cv32e40p.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["cv32e40p"]
        utils.clone_wrapper(repo, srcDir, refresh=rebuild)
    context.cache["cv32e40p.src_dir"] = srcDir


@Tasks.needs(["cv32e40p.src_dir", "corevverif.src_dir", "verilator.install_dir"])
@Tasks.provides(["cv32e40p.verilator_executable", "cv32e40p.build_dir", "cv32e40p.install_dir"])
@Tasks.param("trace", [False, True])
@Tasks.param("waves", [False])
@Tasks.validate(_validate_cv32e40p)
@Tasks.register(category=TaskType.TARGET)
def build_cv32e40p(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build cv32e40p rtl."""
    if not params:
        params = {}
    trace = params.get("trace", False)
    waves = params.get("waves", False)
    user_vars = context.environment.vars
    if "cv32e40p.verilator_executable" in user_vars:  # TODO: also check command line flags?
        return False
    name = utils.makeDirName("cv32e40p")
    cv32e40pDir = Path(context.cache["cv32e40p.src_dir"])
    corevverifDir = Path(context.cache["corevverif.src_dir"])
    verilatorDir = Path(context.cache["verilator.install_dir"])
    cv32e40pSimDir = corevverifDir / "cv32e40p" / "sim" / "core"
    # buildDir = context.environment.paths["deps"].path / "build" / name
    buildDir = cv32e40pSimDir / "cobj_dir"
    installDir = context.environment.paths["deps"].path / "install" / name
    exe = installDir / "verilator_executable"
    if rebuild or not (utils.is_populated(buildDir) and exe.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(buildDir)
        args = [
            "CV_SW_TOOLCHAIN=_",
            "CV_SW_PREFIX=_",
            f"CV_CORE_PATH={cv32e40pDir}",
            f"CV_CORE_PKG={cv32e40pDir}",
            "VERBOSE=1",
        ]
        args.append(f"LOG_INSNS={int(trace)}")
        args.append(f"WAVES={int(waves)}")
        env = os.environ.copy()
        old_path = env["PATH"]
        env["PATH"] = f"{verilatorDir}/bin:{old_path}"
        utils.make("clean", *args, cwd=cv32e40pSimDir, threads=1, live=verbose, env=env)
        utils.make("testbench_verilator", *args, cwd=cv32e40pSimDir, threads=threads, live=verbose, env=env)
        utils.mkdirs(installDir)
        utils.move(cv32e40pSimDir / "simulation_results" / "hello-world" / "verilator_executable", exe)
    context.cache["cv32e40p.build_dir"] = buildDir
    context.cache["cv32e40p.install_dir"] = installDir
    context.cache["cv32e40p.verilator_executable"] = exe
    context.export_paths.add(installDir)
