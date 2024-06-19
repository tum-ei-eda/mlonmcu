import os
import stat
import multiprocessing

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from mlonmcu.setup.tasks.common import get_task_factory, _validate_gcc

logger = get_logger()

Tasks = get_task_factory()


def _validate_riscv_gcc_new(context: MlonMcuContext, params=None):
    if _validate_gcc(context, params=params) and context.environment.has_target("abc"):
        if params:
            vext = params.get("vext", False)
            pext = params.get("pext", False)
        if not vext and not pext:
            return True
    return validate_riscv_gcc_orig(context, params=params)


validate_riscv_gcc_orig = Tasks.validates["install_riscv_gcc"]
Tasks.validates["install_riscv_gcc"] = _validate_riscv_gcc_new


def _validate_abc(context: MlonMcuContext, params=None):
    ret = context.environment.has_target("abc")
    return ret


@Tasks.provides(["abc.src_dir", "abc.sw_dir"])
@Tasks.validate(_validate_abc)
@Tasks.register(category=TaskType.TARGET)
def clone_abc(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the ABC repository."""
    abcName = utils.makeDirName("abc")
    abcSrcDir = context.environment.paths["deps"].path / "src" / abcName
    if rebuild or not utils.is_populated(abcSrcDir):
        abcRepo = context.environment.repos["abc"]
        utils.clone_wrapper(abcRepo, abcSrcDir, refresh=rebuild)
    context.cache["abc.src_dir"] = abcSrcDir
    context.cache["abc.sw_dir"] = abcSrcDir / "sw"


@Tasks.needs(["abc.src_dir"])
@Tasks.provides(["abc.install_dir", "abc.exe"])
@Tasks.validate(_validate_abc)
@Tasks.register(category=TaskType.TARGET)
def build_abc(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Generate the ABC script."""
    abcName = utils.makeDirName("abc")
    # abcSrcDir = context.cache["abc.src_dir"]
    abcInstallDir = context.environment.paths["deps"].path / "install" / abcName
    abcExe = abcInstallDir / "abc.sh"
    if rebuild or not utils.is_populated(abcInstallDir):
        abcInstallDir.mkdir(exist_ok=True)
        with open(abcExe, "w") as f:
            f.write("#!/bin/bash\necho 42\n")

        st = os.stat(abcExe)
        os.chmod(abcExe, st.st_mode | stat.S_IEXEC)
    context.cache["abc.install_dir"] = abcInstallDir
    context.cache["abc.exe"] = abcExe
