from mlonmcu.setup.task import Task, TaskType
from mlonmcu.setup.utils import makeDirName, makeFlags
import logging
from pathlib import Path

logger = logging.getLogger('mlonmcu')
logger.setLevel(logging.DEBUG)

class Context:

    def __init__(self):
        self._vars = {}

    def __setitem__(self, name, value):
        if not isinstance(name, tuple):
            name = (name,frozenset())
        else:
            assert len(name) == 2
            if not isinstance(name[1], frozenset):
                name = (name[0], frozenset(name[1]))
        self._vars[name[0]] = value  # Holds latest value to
        self._vars[name] = value

    def __getitem__(self, name):
        if not isinstance(name, tuple):
            name = (name,frozenset())
        else:
            assert len(name) == 2
            if not isinstance(name[1], frozenset):
                name = (name[0], frozenset(name[1]))
        return self._vars[name]

ctx = Context()


DEPS_DIR = Path("deps")
SRC_DIR = DEPS_DIR / "src"
BUILD_DIR = DEPS_DIR / "build"
INSTALL_DIR = DEPS_DIR / "install"



######
# tf #
######

@Task.provides(["tf.src_dir"])
@Task.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(context, params={}, rebuild=False):
    tfName = makeDirName("tf")
    tfSrcDir = SRC_DIR / tfName
    context["tf.src_dir"] = tfSrcDir


@Task.needs(["tf.src_dir"])
@Task.provides(["tf.dl_dir", "tf.lib_path"])
@Task.param("dbg", False)
@Task.register(category=TaskType.FRAMEWORK)
def build_tensorflow(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    tfName = makeDirName("tf", flags=flags)
    tflmDir = context["tf.src_dir"] / "tensorflow" / "lite" / "micro"
    tflmBuildDir = tflmDir / "tools" / "make"
    tflmDownloadsDir = tflmBuildDir / "downloads"
    context["tf.dl_dir"] = tflmDownloadsDir
    if params["dbg"]:
        tflmLib = tflmBuildDir / "gen" / "lib"
    else:
        tflmLib = tflmBuildDir / "gen" / "lib_dbg"
    context["tf.lib_path", flags] = tflmLib


#########
# tflmc #
#########

@Task.provides(["tflmc.src_dir"])
@Task.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(context, params={}, rebuild=False):
    tflmcName = makeDirName("tflmc")
    tflmcSrcDir = SRC_DIR / tflmcName
    context["tflmc.src_dir"] = tflmcSrcDir

supported_features = ["muriscvnn"]

def _validate_build_tflite_micro_compiler(context, params={}):
    if "muriscvnn" in params:
        if params["muriscvnn"]:
            if "muriscvnn" not in supported_features:
                return False
    return True

@Task.needs(["tflmc.src_dir", "tf.src_dir"])
@Task.optional(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Task.provides(["tflmc.build_dir", "tflmc.exe"])
@Task.param("muriscvnn", [False, True])
@Task.param("dbg", False)
@Task.validate(_validate_build_tflite_micro_compiler)
@Task.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(context, params={}, rebuild=False):
    flags = makeFlags((params["muriscvnn"], "muriscvnn"), (params["dbg"], "dbg"))
    tflmcName = makeDirName("tflmc", flags=flags)
    tflmcBuildDir = BUILD_DIR / tflmcName
    tflmcExe = tflmcBuildDir / "compiler"
    context["tflmc.build_dir", flags] = tflmcBuildDir
    context["tflmc.exe", flags] = tflmcExe


#############
# riscv_gcc #
#############

@Task.provides(["riscv_gcc.install_dir"])
@Task.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(context, params={}, rebuild=False):
    riscvName = makeDirName("riscv_gcc")
    riscvInstallDir = INSTALL_DIR / riscvName
    context["riscv_gcc.install_dir"] = riscvInstallDir


########
# llvm #
########

@Task.provides(["llvm.install_dir"])
@Task.register(category=TaskType.MISC)
def install_llvm(context, params={}, rebuild=False):
    llvmName = makeDirName("llvm")
    llvmInstallDir = INSTALL_DIR / llvmName
    context["llvm.install_dir"] = llvmInstallDir


#########
# etiss #
#########

@Task.provides(["etiss.src_dir"])
@Task.register(category=TaskType.TARGET)
def clone_etiss(context, params={}, rebuild=False):
    etissName = makeDirName("etiss")
    etissSrcDir = SRC_DIR / etissName
    context["etiss.src_dir"] = etissSrcDir


@Task.needs(["etiss.src_dir"])
@Task.provides(["etiss.build_dir"])
@Task.param("dbg", False)
@Task.register(category=TaskType.TARGET)
def build_etiss(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    etissName = makeDirName("etiss", flags=flags)
    etissBuildDir = BUILD_DIR / etissName
    context["etiss.build_dir", flags] = etissBuildDir


@Task.needs(["etiss.src_dir"])
@Task.provides(["etiss.install_dir", "etissvp.src_dir"])
@Task.param("dbg", False)
@Task.register(category=TaskType.TARGET)
def install_etiss(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    etissName = makeDirName("etiss", flags=flags)
    etissBuildDir = context["etiss.build_dir", flags]
    etissInstallDir = INSTALL_DIR / etissName
    context["etiss.install_dir", flags] = etissInstallDir
    etissvpSrcDir = etissInstallDir / "examples" / "bare_etiss_processor"
    context["etissvp.src_dir", flags] = etissvpSrcDir
    # return True


@Task.needs(["etissvp.src_dir"])
@Task.provides(["etissvp.build_dir", "etissvp.exe"])
@Task.param("dbg", False)
@Task.register(category=TaskType.TARGET)
def build_etissvp(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    etissvpName = makeDirName("etissvp", flags=flags)
    etissvpSrcDir = context["etissvp.src_dir", flags]
    etissvpBuildDir = etissvpSrcDir / "build"
    context["etissvp.build_dir", flags] = etissvpBuildDir
    etissvpExe = etissvpBuildDir / "bare_etiss_processor"
    context["etissvp.exe", flags] = etissvpExe


#######
# tvm #
#######

@Task.provides(["tvm.src_dir"])
@Task.register(category=TaskType.FRAMEWORK)
def clone_tvm(context, params={}, rebuild=False):
    tvmName = makeDirName("tvm")
    tvmSrcDir = SRC_DIR / tvmName
    context["tvm.src_dir"] = tvmSrcDir


@Task.needs(["tvm.src_dir"])
@Task.provides(["tvm.build_dir", "tvm.lib", "tvm.pythonpath"])
@Task.param("dbg", False)
@Task.register(category=TaskType.FRAMEWORK)
def build_tvm(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    tvmName = makeDirName("tvm", flags=flags)
    tvmSrcDir = context["tvm.src_dir"]
    tvmBuildDir = BUILD_DIR / tvmName
    context["tvm.build_dir", flags] = tvmBuildDir
    tvmLib = tvmBuildDir / "libtvm.so"
    context["tvm.lib", flags] = tvmLib
    tvmPythonPath = tvmSrcDir / "python"
    context["tvm.pythonpath"] = tvmPythonPath


##########
# utvmcg #
##########

@Task.provides(["utvmcg.src_dir"])
@Task.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(context, params={}, rebuild=False):
    utvmcgName = makeDirName("utvmcg")
    utvmcgSrcDir = SRC_DIR / utvmcgName
    context["utvmcg.src_dir"] = utvmcgSrcDir


@Task.needs(["utvmcg.src_dir", "tvm.src_dir"])
@Task.provides(["utvmcg.build_dir", "utvmcg.exe"])
@Task.param("dbg", False)
@Task.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    utvmcgName = makeDirName("utvmcg", flags=flags)
    utvmcgSrcDir = context["utvmcg.src_dir"]
    utvmcgBuildDir = BUILD_DIR / utvmcgName
    context["utvmcg.build_dir", flags] = utvmcgBuildDir
    utvmcgExe = utvmcgBuildDir / "compiler"
    context["utvmcg.exe", flags] = utvmcgExe


#############
# muriscvnn #
#############

@Task.provides(["muriscvnn.src_dir"])
@Task.register(category=TaskType.OPT)
def clone_muriscvnn(context, params={}, rebuild=False):
    muriscvnnName = makeDirName("muriscvnn")
    muriscvnnSrcDir = SRC_DIR / muriscvnnName
    context["muriscvnn.src_dir"] = muriscvnnSrcDir

@Task.needs(["muriscvnn.src_dir", "etiss.install_dir"])
@Task.provides(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Task.param("dbg", [False, True])
@Task.register(category=TaskType.OPT)
def build_muriscvnn(context, params={}, rebuild=False):
    flags = makeFlags((params["dbg"], "dbg"))
    muriscvnnName = makeDirName("muriscvnn", flags=flags)
    muriscvnnSrcDir = context["muriscvnn.src_dir"]
    muriscvnnBuildDir = BUILD_DIR / muriscvnnName
    context["muriscvnn.build_dir", flags] = muriscvnnSrcDir
    muriscvnnIncludeDir = muriscvnnBuildDir / "Includes"
    context["muriscvnn.inc_dir", flags] = muriscvnnIncludeDir

# WIP:

# TODO:
supported_frameworks = ["tflm", "tvm"]
enabled_frameworks = supported_frameworks
supported_backends = ["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg"]
enabled_backends = supported_backends
supported_backend_per_framework = {"tflm": ["tflmc", "tflmi"], "tvm": ["tvmaot", "tvmrt", "tvmcg"]}

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


@Task.optional(["tf.src_dir", "tvm.src_dir", "muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Task.provides(["mlif.build_dir", "mlif.lib_path"])
@Task.param("framework", ["tflm", "tvm"])  # TODO: from context?
@Task.param("backend", ["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg"])  # TODO: from context?
@Task.param("muriscvnn", [False, True])
@Task.param("dbg", [False, True])
@Task.validate(_validate_build_milf)
@Task.register(category=TaskType.OPT)
def build_mlif(context, params={}, rebuild=False):
    flags = makeFlags((True, params["backend"]),(params["dbg"], "dbg"))  # params["framework"] not required?
    mlifName = makeDirName("mlif", flags=flags)
    mlifSrc = Path("sw") / "lib" / "ml_interface"
    mlifBuildDir = BUILD_DIR / mlifName
    context["mlif.build_dir", flags] = mlifBuildDir
    mlifLib = mlifBuildDir / "libmlif.a"
    context["mlif.lib_path", flags] = mlifLib

# TODO: return True if changed, False if not, or: reurn code: unchanged, changed, error, unknown

def install_dependencies():
    Task.reset_changes()
    print("registry", Task.registry)
    print("dependencies", Task.dependencies)
    print("providers", Task.providers)
    V, E = Task.get_graph()
    print("(V, E)", (V, E))
    order = Task.get_order()
    logger.debug("Determined dependency order: %s" % str(order))

    for task in order:
        func = Task.registry[task]
        func(ctx)

    print(ctx._vars)

# install_dependencies()
