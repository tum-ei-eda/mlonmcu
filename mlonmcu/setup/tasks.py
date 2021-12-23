from mlonmcu.setup.task import Task, TaskType
import logging
from pathlib import Path
import time
from mlonmcu.context import MlonMcuContext
import mlonmcu.setup.utils as utils
from git import Repo

logger = logging.getLogger('mlonmcu')
logger.setLevel(logging.DEBUG)


# WIP:

# TODO:
supported_frameworks = ["tflm", "tvm"]
enabled_frameworks = supported_frameworks
supported_backends = ["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg"]
enabled_backends = supported_backends
supported_backend_per_framework = {"tflm": ["tflmc", "tflmi"], "tvm": ["tvmaot", "tvmrt", "tvmcg"]}
supported_features = ["muriscvnn"]
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


# DEPS_DIR = Path("deps")
# SRC_DIR = DEPS_DIR / "src"
# BUILD_DIR = DEPS_DIR / "build"
# INSTALL_DIR = DEPS_DIR / "install"



######
# tf #
######

def _validate_tensorflow(context, params={}):
    if "tflm" not in supported_frameworks:
        return False
    return True

@Task.provides(["tf.src_dir"])
@Task.validate(_validate_tensorflow)
@Task.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(context, params={}, rebuild=False):
    tfName = utils.makeDirName("tf")
    tfSrcDir = context.environment.paths["deps"].path / "src" / tfName
    if rebuild or not tfSrcDir.is_dir():
        tfRepo = context.environment.repos["tensorflow"]
        if tfRepo.ref:
            Repo.clone_from(tfRepo.url, tfSrcDir, branch=tfRepo.ref)
        else:
            Repo.clone_from(tfRepo.url, tfSrcDir)
    context.cache["tf.src_dir"] = tfSrcDir


@Task.needs(["tf.src_dir"])
@Task.provides(["tf.dl_dir", "tf.lib_path"])
#@Task.param("dbg", False)
@Task.param("dbg", True)
@Task.validate(_validate_tensorflow)
@Task.register(category=TaskType.FRAMEWORK)
def build_tensorflow(context, params={}, rebuild=False):
    print("params", params)
    flags = utils.makeFlags((params["dbg"], "dbg"))
    tfName = utils.makeDirName("tf", flags=flags)
    tfSrcDir = context.cache["tf.src_dir"]
    tflmDir = tfSrcDir / "tensorflow" / "lite" / "micro"
    tflmBuildDir = tflmDir / "tools" / "make"
    tflmDownloadsDir = tflmBuildDir / "downloads"
    if params["dbg"]:
        tflmLib = tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"  # FIXME: add _dbg suffix is possible
    else:
        tflmLib = tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
    if rebuild or not tflmLib.is_file() or not tflmDownloadsDir.is_dir():
        tfDbgArg = ["BUILD_TYPE=debug"] if params["dbg"] else []
        utils.make("-f", str(tflmDir / "tools" / "make" / "Makefile"), "hello_world_bin", *tfDbgArg, cwd=tfSrcDir)
    context.cache["tf.dl_dir"] = tflmDownloadsDir
    context.cache["tf.lib_path", flags] = tflmLib


#########
# tflmc #
#########


def _validate_tflite_micro_compiler(context, params={}):
    if not _validate_tensorflow(context, params=params):
        return False
    if "tflmc" not in enabled_backends:
        return False
    return True


@Task.provides(["tflmc.src_dir"])
@Task.validate(_validate_tflite_micro_compiler)
@Task.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(context, params={}, rebuild=False):
    tflmcName = utils.makeDirName("tflmc")
    tflmcSrcDir = context.environment.paths["deps"].path / "src" / tflmcName
    if rebuild or not tflmcSrcDir.is_dir():
        tflmcRepo = context.environment.repos["tflite_micro_compiler"]
        if tflmcRepo.ref:
            Repo.clone_from(tflmcRepo.url, tflmcSrcDir, branch=tflmcRepo.ref)
        else:
            Repo.clone_from(tflmcRepo.url, tflmcSrcDir)
    context.cache["tflmc.src_dir"] = tflmcSrcDir

def _validate_build_tflite_micro_compiler(context, params={}):
    if "muriscvnn" in params:
        if params["muriscvnn"]:
            if "muriscvnn" not in supported_features:
                return False
    return _validate_tflite_micro_compiler(context, params=params)

@Task.needs(["tflmc.src_dir", "tf.src_dir"])
@Task.optional(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Task.provides(["tflmc.build_dir", "tflmc.exe"])
@Task.param("muriscvnn", [False, True])
@Task.param("dbg", False)
@Task.validate(_validate_build_tflite_micro_compiler)
@Task.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(context, params={}, rebuild=False):
    flags = utils.makeFlags((params["muriscvnn"], "muriscvnn"), (params["dbg"], "dbg"))
    flags_ = utils.makeFlags((params["dbg"], "dbg"))
    tflmcName = utils.makeDirName("tflmc", flags=flags)
    tflmcBuildDir = context.environment.paths["deps"].path / "build" / tflmcName
    tflmcExe = tflmcBuildDir / "compiler"
    tfSrcDir = context.cache["tf.src_dir", flags_]
    tflmcSrcDir = context.cache["tflmc.src_dir", flags_]
    if rebuild or not tflmcBuildDir.is_dir() or not tflmcExe.is_file():
        utils.mkdirs(tflmcBuildDir)
        utils.cmake("-DTF_DIR=" + str(tfSrcDir), str(tflmcSrcDir), debug=params["dbg"], cwd=tflmcBuildDir)
        utils.make(cwd=tflmcBuildDir)
    context.cache["tflmc.build_dir", flags] = tflmcBuildDir
    context.cache["tflmc.exe", flags] = tflmcExe


#############
# riscv_gcc #
#############

def _validate_riscv_gcc(context, params={}):
    if "etiss" not in enabled_targets:
        return False
    return True

@Task.provides(["riscv_gcc.install_dir"])
@Task.validate(_validate_riscv_gcc)
@Task.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(context, params={}, rebuild=False):
    riscvName = utils.makeDirName("riscv_gcc")
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    context.cache["riscv_gcc.install_dir"] = riscvInstallDir


########
# llvm #
########

def _validate_llvm(context, params={}):
    if "tvm" not in enabled_frameworks:
        return False
    return True

@Task.provides(["llvm.install_dir"])
@Task.validate(_validate_llvm)
@Task.register(category=TaskType.MISC)
def install_llvm(context, params={}, rebuild=False):
    llvmName = utils.makeDirName("llvm")
    llvmInstallDir = context.environment.paths["deps"].path / "install" / llvmName
    context.cache["llvm.install_dir"] = llvmInstallDir


#########
# etiss #
#########

def _validate_etiss(context, params={}):
    if "etiss" not in enabled_targets:
        return False
    return True

@Task.provides(["etiss.src_dir"])
@Task.validate(_validate_etiss)
@Task.register(category=TaskType.TARGET)
def clone_etiss(context, params={}, rebuild=False):
    etissName = utils.makeDirName("etiss")
    etissSrcDir = context.environment.paths["deps"].path / "src" / etissName
    context.cache["etiss.src_dir"] = etissSrcDir


@Task.needs(["etiss.src_dir"])
@Task.provides(["etiss.build_dir"])
@Task.param("dbg", False)
@Task.validate(_validate_etiss)
@Task.register(category=TaskType.TARGET)
def build_etiss(context, params={}, rebuild=False):
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.environment.paths["deps"].path / "build" / etissName
    context.cache["etiss.build_dir", flags] = etissBuildDir


@Task.needs(["etiss.src_dir"])
@Task.provides(["etiss.install_dir", "etissvp.src_dir"])
@Task.param("dbg", False)
@Task.validate(_validate_etiss)
@Task.register(category=TaskType.TARGET)
def install_etiss(context, params={}, rebuild=False):
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.cache["etiss.build_dir", flags]
    etissInstallDir = context.environment.paths["deps"].path / "install" / etissName
    context.cache["etiss.install_dir", flags] = etissInstallDir
    etissvpSrcDir = etissInstallDir / "examples" / "bare_etiss_processor"
    context.cache["etissvp.src_dir", flags] = etissvpSrcDir
    # return True


@Task.needs(["etissvp.src_dir"])
@Task.provides(["etissvp.build_dir", "etissvp.exe"])
@Task.param("dbg", False)
@Task.validate(_validate_etiss)
@Task.register(category=TaskType.TARGET)
def build_etissvp(context, params={}, rebuild=False):
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

def _validate_tvm(context, params={}):
    if "tvm" not in supported_frameworks:
        return False
    return True

@Task.provides(["tvm.src_dir"])
@Task.validate(_validate_tvm)
@Task.register(category=TaskType.FRAMEWORK)
def clone_tvm(context, params={}, rebuild=False):
    tvmName = utils.makeDirName("tvm")
    tvmSrcDir = context.environment.paths["deps"].path / "install" / tvmName
    context.cache["tvm.src_dir"] = tvmSrcDir


@Task.needs(["tvm.src_dir", "llvm.install_dir"])
@Task.provides(["tvm.build_dir", "tvm.lib", "tvm.pythonpath"])
@Task.param("dbg", False)
@Task.validate(_validate_tvm)
@Task.register(category=TaskType.FRAMEWORK)
def build_tvm(context, params={}, rebuild=False):
    flags = utils.makeFlags((params["dbg"], "dbg"))
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.cache["tvm.src_dir"]
    tvmBuildDir =  tvmSrcDir / "build"
    context.cache["tvm.build_dir", flags] = tvmBuildDir
    tvmLib = tvmBuildDir / "libtvm.so"
    context.cache["tvm.lib", flags] = tvmLib
    tvmPythonPath = tvmSrcDir / "python"
    context.cache["tvm.pythonpath"] = tvmPythonPath


##########
# utvmcg #
##########

def _validate_utvmcg(context, params={}):
    if not _validate_tvm(context, params=params):
        return False
    if "tvmcg" not in supported_backends:
        return False
    return True

@Task.provides(["utvmcg.src_dir"])
@Task.validate(_validate_utvmcg)
@Task.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(context, params={}, rebuild=False):
    utvmcgName = utils.makeDirName("utvmcg")
    utvmcgSrcDir = context.environment.paths["deps"].path / "src" / utvmcgName
    context.cache["utvmcg.src_dir"] = utvmcgSrcDir


@Task.needs(["utvmcg.src_dir", "tvm.src_dir"])
@Task.provides(["utvmcg.build_dir", "utvmcg.exe"])
@Task.param("dbg", False)
# @Task.validate(_validate_utvmcg)
@Task.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(context, params={}, rebuild=False):
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

def _validate_muriscvnn(context, params={}):
    if "muriscvnn" not in supported_features:
        return False
    return True

@Task.provides(["muriscvnn.src_dir"])
@Task.validate(_validate_muriscvnn)
@Task.register(category=TaskType.OPT)
def clone_muriscvnn(context, params={}, rebuild=False):
    muriscvnnName = utils.makeDirName("muriscvnn")
    muriscvnnSrcDir = context.environment.paths["deps"].path / "src" / muriscvnnName
    context.cache["muriscvnn.src_dir"] = muriscvnnSrcDir

@Task.needs(["muriscvnn.src_dir", "etiss.install_dir"])
@Task.provides(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Task.param("dbg", [False, True])
@Task.validate(_validate_muriscvnn)
@Task.register(category=TaskType.OPT)
def build_muriscvnn(context, params={}, rebuild=False):
    flags = utils.makeFlags((params["dbg"], "dbg"))
    muriscvnnName = utils.makeDirName("muriscvnn", flags=flags)
    muriscvnnSrcDir = context.cache["muriscvnn.src_dir"]
    muriscvnnBuildDir = context.environment.paths["deps"].path / "build" / muriscvnnName
    context.cache["muriscvnn.build_dir", flags] = muriscvnnSrcDir
    muriscvnnIncludeDir = muriscvnnBuildDir / "Includes"
    context.cache["muriscvnn.inc_dir", flags] = muriscvnnIncludeDir

def _validate_build_milf(context, params={}):
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


# @Task.optional(["tf.src_dir", "tvm.src_dir", "muriscvnn.build_dir", "muriscvnn.inc_dir"])
# @Task.provides(["mlif.build_dir", "mlif.lib_path"])
# @Task.param("framework", ["tflm", "tvm"])  # TODO: from context?
# @Task.param("backend", ["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg"])  # TODO: from context?
# @Task.param("muriscvnn", [False, True])
# @Task.param("dbg", [False, True])
# @Task.validate(_validate_build_milf)
# @Task.register(category=TaskType.OPT)
# def build_mlif(context, params={}, rebuild=False):
#     flags = utils.makeFlags((True, params["backend"]),(params["dbg"], "dbg"))  # params["framework"] not required?
#     mlifName = utils.makeDirName("mlif", flags=flags)
#     mlifSrc = Path("sw") / "lib" / "ml_interface"
#     mlifBuildDir = context.environment.paths["deps"].path / "build" / mlifName
#     context.cache["mlif.build_dir", flags] = mlifBuildDir
#     mlifLib = mlifBuildDir / "libmlif.a"
#     context.cache["mlif.lib_path", flags] = mlifLib

# TODO: return True if changed, False if not, or: reurn code: unchanged, changed, error, unknown

def install_dependencies(context, progress=False):
    print("context.environment", context.environment)
    print("Task.params", Task.params)
    Task.reset_changes()
    # print("registry", Task.registry)
    # print("dependencies", Task.dependencies)
    # print("providers", Task.providers)
    V, E = Task.get_graph()
    # print("(V, E)", (V, E))
    order = Task.get_order()
    logger.debug("Determined dependency order: %s" % str(order))

    # skip = 1
    # print("num tasks:", len(Task.registry))
    # print("num skip:", skip)
    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(Task.registry), desc="Installing dependencies", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    else:
        pbar = None
    for task in order:
        func = Task.registry[task]
        func(context, progress=progress)
        time.sleep(0.1)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()
    cache_file = context.environment.paths["deps"].path / "cache.ini"
    context.cache.write_to_file(cache_file)

    # print(ctx._vars)

# install_dependencies()
