"""Definition of tasks used to dynamically install MLonMCU dependencies"""

import os
import logging
import tempfile
from urllib.request import urlretrieve
from git import Repo

from mlonmcu.setup.task import TaskFactory, TaskType
from mlonmcu.context import MlonMcuContext
from mlonmcu.setup import utils

logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

Tasks = TaskFactory()

# WIP:

# TODO:
supported_frameworks = ["tflm", "tvm"]
enabled_frameworks = supported_frameworks
supported_backends = ["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg"]
enabled_backends = supported_backends
supported_backend_per_framework = {
    "tflm": ["tflmc", "tflmi"],
    "tvm": ["tvmaot", "tvmrt", "tvmcg"],
}
# supported_features = ["muriscvnn"]
supported_features = []
enabled_targets = ["etiss"]
# class TaskCache:
# class Context:
#
#     def __init__(self):
#         self._vars = {}
#
#     def __setitem__(self, name, value):
#         if not isinstance(name, tuple):
#             name = (name,frozenset())
#         else:
#             assert len(name) == 2
#             if not isinstance(name[1], frozenset):
#                 name = (name[0], frozenset(name[1]))
#         self._vars[name[0]] = value  # Holds latest value
#         self._vars[name] = value
#
#     def __getitem__(self, name):
#         if not isinstance(name, tuple):
#             name = (name,frozenset())
#         else:
#             assert len(name) == 2
#             if not isinstance(name[1], frozenset):
#                 name = (name[0], frozenset(name[1]))
#         return self._vars[name]

# ctx = MlonMcuContext()


######
# tf #
######


def _validate_tensorflow(context : MlonMcuContext, params=None):
    if "tflm" not in supported_frameworks:
        return False
    return True


@Tasks.provides(["tf.src_dir"])
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(context : MlonMcuContext, params=None, rebuild=False):
    """Clone the TF/TFLM repository."""
    tfName = utils.makeDirName("tf")
    tfSrcDir = context.environment.paths["deps"].path / "src" / tfName
    if rebuild or not tfSrcDir.is_dir():
        tfRepo = context.environment.repos["tensorflow"]
        if tfRepo.ref:
            Repo.clone_from(tfRepo.url, tfSrcDir, branch=tfRepo.ref)
        else:
            Repo.clone_from(tfRepo.url, tfSrcDir)
    context.cache["tf.src_dir"] = tfSrcDir


@Tasks.needs(["tf.src_dir"])
@Tasks.provides(["tf.dl_dir", "tf.lib_path"])
# @Tasks.param("dbg", False)
@Tasks.param("dbg", True)
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tensorflow(context : MlonMcuContext, params=None, rebuild=False):
    """Download tensorflow dependencies and build lib."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    tfName = utils.makeDirName("tf", flags=flags)
    tfSrcDir = context.cache["tf.src_dir"]
    tflmDir = tfSrcDir / "tensorflow" / "lite" / "micro"
    tflmBuildDir = tflmDir / "tools" / "make"
    tflmDownloadsDir = tflmBuildDir / "downloads"
    if params["dbg"]:
        tflmLib = (
            tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
        )  # FIXME: add _dbg suffix is possible
    else:
        tflmLib = (
            tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
        )
    # if rebuild or not tflmLib.is_file() or not tflmDownloadsDir.is_dir():
    if rebuild or not tflmDownloadsDir.is_dir():
        tfDbgArg = ["BUILD_TYPE=debug"] if params["dbg"] else []
        utils.make(
            "-f",
            str(tflmDir / "tools" / "make" / "Makefile"),
            "third_party_downloads",
            *tfDbgArg,
            cwd=tfSrcDir,
        )
    context.cache["tf.dl_dir"] = tflmDownloadsDir
    context.cache["tf.lib_path", flags] = tflmLib  # ignore!


#########
# tflmc #
#########


def _validate_tflite_micro_compiler(context : MlonMcuContext, params=None):
    if not _validate_tensorflow(context, params=params):
        return False
    if "tflmc" not in enabled_backends:
        return False
    return True


@Tasks.provides(["tflmc.src_dir"])
@Tasks.validate(_validate_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(context : MlonMcuContext, params=None, rebuild=False):
    """Clone the preinterpreter repository."""
    tflmcName = utils.makeDirName("tflmc")
    tflmcSrcDir = context.environment.paths["deps"].path / "src" / tflmcName
    if rebuild or not tflmcSrcDir.is_dir():
        tflmcRepo = context.environment.repos["tflite_micro_compiler"]
        if tflmcRepo.ref:
            Repo.clone_from(tflmcRepo.url, tflmcSrcDir, branch=tflmcRepo.ref)
        else:
            Repo.clone_from(tflmcRepo.url, tflmcSrcDir)
    context.cache["tflmc.src_dir"] = tflmcSrcDir


def _validate_build_tflite_micro_compiler(context : MlonMcuContext, params=None):
    if params:
        if "muriscvnn" in params:
            if params["muriscvnn"]:
                if "muriscvnn" not in supported_features:
                    return False
    return _validate_tflite_micro_compiler(context, params=params)


@Tasks.needs(["tflmc.src_dir", "tf.src_dir"])
@Tasks.optional(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Tasks.provides(["tflmc.build_dir", "tflmc.exe"])
@Tasks.param("muriscvnn", [False, True])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_build_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(context : MlonMcuContext, params=None, rebuild=False):
    """Build the TFLM preinterpreter."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["muriscvnn"], "muriscvnn"), (params["dbg"], "dbg"))
    flags_ = utils.makeFlags((params["dbg"], "dbg"))
    tflmcName = utils.makeDirName("tflmc", flags=flags)
    tflmcBuildDir = context.environment.paths["deps"].path / "build" / tflmcName
    tflmcInstallDir = context.environment.paths["deps"].path / "install" / tflmcName
    tflmcExe = tflmcInstallDir / "compiler"
    tfSrcDir = context.cache["tf.src_dir", flags_]
    tflmcSrcDir = context.cache["tflmc.src_dir", flags_]
    if rebuild or not tflmcBuildDir.is_dir() or not tflmcExe.is_file():
        utils.mkdirs(tflmcBuildDir)
        # utils.cmake("-DTF_SRC=" + str(tfSrcDir), str(tflmcSrcDir), debug=params["dbg"], cwd=tflmcBuildDir)
        utils.cmake(
            "-DTF_SRC=" + str(tfSrcDir),
            "-DGET_TF_SRC=ON",
            str(tflmcSrcDir),
            debug=params["dbg"],
            cwd=tflmcBuildDir,
            live=True,
        )
        utils.make(cwd=tflmcBuildDir)
        utils.move(tflmcBuildDir / "compiler", tflmcExe)  # TODO: os.rename
    context.cache["tflmc.build_dir", flags] = tflmcBuildDir
    context.cache["tflmc.exe", flags] = tflmcExe


#############
# riscv_gcc #
#############


def _validate_riscv_gcc(context : MlonMcuContext, params=None):
    if "etiss" not in enabled_targets:
        return False
    return True


@Tasks.provides(["riscv_gcc.install_dir"])
@Tasks.validate(_validate_riscv_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(context : MlonMcuContext, params=None, rebuild=False):
    """Download and install the RISCV GCC toolchain."""
    riscvName = utils.makeDirName("riscv_gcc")
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    user_vars = context.environment.vars
    if "riscv_gcc.dir" in user_vars:
        # TODO: WARNING
        riscvInstallDir = user_vars["riscv_gcc.dir"]
    else:
        riscvVersion = (
            user_vars["riscv.version"]
            if "riscv.version" in user_vars
            else "8.3.0-2020.04.0"
        )
        riscvDist = (
            user_vars["riscv.distribution"]
            if "riscv.distribution" in user_vars
            else "x86_64-linux-ubuntu14"
        )
        riscvUrl = "https://static.dev.sifive.com/dev-tools/"
        riscvFileName = f"riscv64-unknown-elf-gcc-{riscvVersion}-{riscvDist}"
        riscvArchive = riscvFileName + ".tar.gz"
        if (
            rebuild
            or not riscvInstallDir.is_dir()
            or not os.listdir(riscvInstallDir.resolve())
        ):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmpArchive = os.path.join(tmp_dir, riscvArchive)
                urlretrieve(
                    riscvUrl + riscvArchive, tmpArchive
                )  # TODO: replace by exec(wget)?
                utils.exec("tar", "xf", tmpArchive, cwd=tmp_dir)
                os.remove(
                    os.path.join(tmp_dir, tmpArchive)
                )  # Cleanup in tmpdir not neccessary
                utils.mkdirs(riscvInstallDir)
                os.rename(os.path.join(tmp_dir, riscvFileName), riscvInstallDir)
    context.cache["riscv_gcc.install_dir"] = riscvInstallDir


########
# llvm #
########


def _validate_llvm(context : MlonMcuContext, params=None):
    if "tvm" not in enabled_frameworks:
        return False
    return True


@Tasks.provides(["llvm.install_dir"])
@Tasks.validate(_validate_llvm)
@Tasks.register(category=TaskType.MISC)
def install_llvm(context : MlonMcuContext, params=None, rebuild=False):
    """Download and install LLVM."""
    llvmName = utils.makeDirName("llvm")
    llvmInstallDir = context.environment.paths["deps"].path / "install" / llvmName
    user_vars = context.environment.vars
    llvmVersion = user_vars["llvm.version"] if "llvm.version" in user_vars else "11.0.1"
    llvmDist = (
        user_vars["llvm.distribution"]
        if "llvm.distribution" in user_vars
        else "x86_64-linux-gnu-ubuntu-16.04"
    )
    llvmUrl = (
        f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvmVersion}/"
    )
    llvmFileName = f"clang+llvm-{llvmVersion}-{llvmDist}"
    llvmArchive = llvmFileName + ".tar.xz"
    if (
        rebuild
        or not llvmInstallDir.is_dir()
        or not os.listdir(llvmInstallDir.resolve())
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmpArchive = os.path.join(tmp_dir, llvmArchive)
            urlretrieve(
                llvmUrl + llvmArchive, tmpArchive
            )  # TODO: replace by exec(wget)?
            utils.exec("tar", "xf", tmpArchive, cwd=tmp_dir)
            os.remove(
                os.path.join(tmp_dir, tmpArchive)
            )  # Cleanup in tmpdir not neccessary
            utils.mkdirs(llvmInstallDir)
            os.rename(os.path.join(tmp_dir, llvmFileName), llvmInstallDir)
    context.cache["llvm.install_dir"] = llvmInstallDir


#########
# etiss #
#########


def _validate_etiss(context : MlonMcuContext, params={}):
    if "etiss" not in enabled_targets:
        return False
    return True


@Tasks.provides(["etiss.src_dir"])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def clone_etiss(context : MlonMcuContext, params=None, rebuild=False):
    """Clone the ETISS repository."""
    etissName = utils.makeDirName("etiss")
    etissSrcDir = context.environment.paths["deps"].path / "src" / etissName
    if rebuild or not etissSrcDir.is_dir():
        etissRepo = context.environment.repos["etiss"]
        if etissRepo.ref:
            Repo.clone_from(etissRepo.url, etissSrcDir, branch=etissRepo.ref)
        else:
            Repo.clone_from(etissRepo.url, etissSrcDir)
    context.cache["etiss.src_dir"] = etissSrcDir


@Tasks.needs(["etiss.src_dir"])
@Tasks.provides(["etiss.build_dir"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etiss(context : MlonMcuContext, params=None, rebuild=False):
    """Build the ETISS simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.environment.paths["deps"].path / "build" / etissName
    context.cache["etiss.build_dir", flags] = etissBuildDir


@Tasks.needs(["etiss.src_dir"])
@Tasks.provides(["etiss.install_dir", "etissvp.src_dir"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def install_etiss(context : MlonMcuContext, params=None, rebuild=False):
    """Install ETISS."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.cache["etiss.build_dir", flags]
    etissInstallDir = context.environment.paths["deps"].path / "install" / etissName
    context.cache["etiss.install_dir", flags] = etissInstallDir
    etissvpSrcDir = etissInstallDir / "examples" / "bare_etiss_processor"
    context.cache["etissvp.src_dir", flags] = etissvpSrcDir
    # return True


@Tasks.needs(["etissvp.src_dir"])
@Tasks.provides(["etissvp.build_dir", "etissvp.exe"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etissvp(context : MlonMcuContext, params=None, rebuild=False):
    """Build the ETISS virtual prototype."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissvpName = utils.makeDirName("etissvp", flags=flags)
    etissvpSrcDir = context.cache["etissvp.src_dir", flags]
    etissvpBuildDir = etissvpSrcDir / "build"
    context.cache["etissvp.build_dir", flags] = etissvpBuildDir
    etissvpExe = etissvpBuildDir / "bare_etiss_processor"
    context.cache["etissvp.exe", flags] = etissvpExe


#######
# tvm #
#######


def _validate_tvm(context : MlonMcuContext, params=None):
    if "tvm" not in supported_frameworks:
        return False
    return True


@Tasks.provides(["tvm.src_dir"])
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tvm(context : MlonMcuContext, params=None, rebuild=False):
    """Clone the TVM repository."""
    tvmName = utils.makeDirName("tvm")
    tvmSrcDir = context.environment.paths["deps"].path / "install" / tvmName
    if rebuild or not tvmSrcDir.is_dir():
        tvmRepo = context.environment.repos["tvm"]
        if tvmRepo.ref:
            Repo.clone_from(tvmRepo.url, tvmSrcDir, branch=tvmRepo.ref)
        else:
            Repo.clone_from(tvmRepo.url, tvmSrcDir)
    context.cache["tvm.src_dir"] = tvmSrcDir


@Tasks.needs(["tvm.src_dir", "llvm.install_dir"])
@Tasks.provides(["tvm.build_dir", "tvm.lib", "tvm.pythonpath"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tvm(context : MlonMcuContext, params=None, rebuild=False):
    """Build the TVM framework."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.cache["tvm.src_dir"]
    tvmBuildDir = tvmSrcDir / "build"
    context.cache["tvm.build_dir", flags] = tvmBuildDir
    tvmLib = tvmBuildDir / "libtvm.so"
    context.cache["tvm.lib", flags] = tvmLib
    tvmPythonPath = tvmSrcDir / "python"
    context.cache["tvm.pythonpath"] = tvmPythonPath


##########
# utvmcg #
##########


def _validate_utvmcg(context : MlonMcuContext, params=None):
    if not _validate_tvm(context, params=params):
        return False
    if "tvmcg" not in supported_backends:
        return False
    return True


@Tasks.provides(["utvmcg.src_dir"])
@Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(context : MlonMcuContext, params=None, rebuild=False):
    """Clone the uTVM code generator."""
    utvmcgName = utils.makeDirName("utvmcg")
    utvmcgSrcDir = context.environment.paths["deps"].path / "src" / utvmcgName
    if rebuild or not utvmcgSrcDir.is_dir():
        utvmcgRepo = context.environment.repos["utvm_staticrt_codegen"]
        if utvmcgRepo.ref:
            Repo.clone_from(utvmcgRepo.url, utvmcgSrcDir, branch=utvmcgRepo.ref)
        else:
            Repo.clone_from(utvmcgRepo.url, utvmcgSrcDir)
    context.cache["utvmcg.src_dir"] = utvmcgSrcDir


@Tasks.needs(["utvmcg.src_dir", "tvm.src_dir"])
@Tasks.provides(["utvmcg.build_dir", "utvmcg.exe"])
@Tasks.param("dbg", False)
# @Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(context : MlonMcuContext, params=None, rebuild=False):
    """Build the uTVM code generator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    utvmcgName = utils.makeDirName("utvmcg", flags=flags)
    utvmcgSrcDir = context.cache["utvmcg.src_dir"]
    utvmcgBuildDir = context.environment.paths["deps"].path / "build" / utvmcgName
    context.cache["utvmcg.build_dir", flags] = utvmcgBuildDir
    utvmcgExe = utvmcgBuildDir / "compiler"
    context.cache["utvmcg.exe", flags] = utvmcgExe


#############
# muriscvnn #
#############


def _validate_muriscvnn(context : MlonMcuContext, params=None):
    if "muriscvnn" not in supported_features:
        return False
    return True


@Tasks.provides(["muriscvnn.src_dir"])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def clone_muriscvnn(context : MlonMcuContext, params=None, rebuild=False):
    """Clone the muRISCV-NN project."""
    muriscvnnName = utils.makeDirName("muriscvnn")
    muriscvnnSrcDir = context.environment.paths["deps"].path / "src" / muriscvnnName
    if rebuild or not muriscvnnSrcDir.is_dir():
        muriscvnnRepo = context.environment.repos["muriscvnn"]
        if muriscvnnRepo.ref:
            Repo.clone_from(
                muriscvnnRepo.url, muriscvnnSrcDir, branch=muriscvnnRepo.ref
            )
        else:
            Repo.clone_from(muriscvnnRepo.url, muriscvnnSrcDir)
    context.cache["muriscvnn.src_dir"] = muriscvnnSrcDir


@Tasks.needs(["muriscvnn.src_dir", "etiss.install_dir"])
@Tasks.provides(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def build_muriscvnn(context : MlonMcuContext, params=None, rebuild=False):
    """Build muRISCV-NN."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    muriscvnnName = utils.makeDirName("muriscvnn", flags=flags)
    muriscvnnSrcDir = context.cache["muriscvnn.src_dir"]
    muriscvnnBuildDir = context.environment.paths["deps"].path / "build" / muriscvnnName
    context.cache["muriscvnn.build_dir", flags] = muriscvnnSrcDir
    muriscvnnIncludeDir = muriscvnnBuildDir / "Includes"
    context.cache["muriscvnn.inc_dir", flags] = muriscvnnIncludeDir


def _validate_build_milf(context : MlonMcuContext, params=None):
    if not params:
        params = {}
    assert "framework" in params, "Missing param: framework"
    framework = params["framework"]
    assert framework in supported_frameworks, f"Unsupported Configuration: {params}"
    if framework not in enabled_frameworks:
        return False  # Skip
    assert "backend" in params, "Missing param: backend"
    backend = params["backend"]
    assert backend in supported_backends, f"Unsupported Configuration: {params}"
    if backend not in supported_backend_per_framework[framework]:
        return False  # Skip
    if backend not in enabled_backends:
        return False  # Skip
    if "muriscvnn" in params:
        if params["muriscvnn"]:
            if framework != "tflm":
                return False  # Skip
    return True


# @Tasks.optional(["tf.src_dir", "tvm.src_dir", "muriscvnn.build_dir", "muriscvnn.inc_dir"])
# @Tasks.provides(["mlif.build_dir", "mlif.lib_path"])
# @Tasks.param("framework", ["tflm", "tvm"])  # TODO: from context?
# @Tasks.param("backend", ["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg"])  # TODO: from context?
# @Tasks.param("muriscvnn", [False, True])
# @Tasks.param("dbg", [False, True])
# @Tasks.validate(_validate_build_milf)
# @Tasks.register(category=TaskType.OPT)
# def build_mlif(context, params={}, rebuild=False):
#     flags = utils.makeFlags((True, params["backend"]),(params["dbg"], "dbg"))  # params["framework"] not required?
#     mlifName = utils.makeDirName("mlif", flags=flags)
#     mlifSrc = Path("sw") / "lib" / "ml_interface"
#     mlifBuildDir = context.environment.paths["deps"].path / "build" / mlifName
#     context.cache["mlif.build_dir", flags] = mlifBuildDir
#     mlifLib = mlifBuildDir / "libmlif.a"
#     context.cache["mlif.lib_path", flags] = mlifLib

# TODO: return True if changed, False if not, or: reurn code: unchanged, changed, error, unknown

# install_dependencies()
