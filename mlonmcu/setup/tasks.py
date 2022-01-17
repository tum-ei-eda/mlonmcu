"""Definition of tasks used to dynamically install MLonMCU dependencies"""

import os
import logging
from pathlib import Path

from mlonmcu.setup.task import TaskFactory, TaskType
from mlonmcu.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

logger = get_logger()

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
supported_features = ["muriscvnn", "vext"]
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


def _validate_tensorflow(context: MlonMcuContext, params=None):
    if "tflm" not in supported_frameworks:
        return False
    return True


@Tasks.provides(["tf.src_dir"])
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(context: MlonMcuContext, params=None, rebuild=False):
    """Clone the TF/TFLM repository."""
    tfName = utils.makeDirName("tf")
    tfSrcDir = context.environment.paths["deps"].path / "src" / tfName
    if rebuild or not utils.is_populated(tfSrcDir):
        tfRepo = context.environment.repos["tensorflow"]
        utils.clone(tfRepo.url, tfSrcDir, branch=tfRepo.ref)
    context.cache["tf.src_dir"] = tfSrcDir


@Tasks.needs(["tf.src_dir"])
@Tasks.provides(["tf.dl_dir", "tf.lib_path"])
# @Tasks.param("dbg", False)
@Tasks.param("dbg", True)
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tensorflow(context: MlonMcuContext, params=None, rebuild=False):
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
    # if rebuild or not tflmLib.is_file() or not utils.is_populated(tflmDownloadsDir):
    if rebuild or not utils.is_populated(tflmDownloadsDir):
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


def _validate_tflite_micro_compiler(context: MlonMcuContext, params=None):
    if not _validate_tensorflow(context, params=params):
        return False
    if "tflmc" not in enabled_backends:
        return False
    return True


@Tasks.provides(["tflmc.src_dir"])
@Tasks.validate(_validate_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(context: MlonMcuContext, params=None, rebuild=False):
    """Clone the preinterpreter repository."""
    tflmcName = utils.makeDirName("tflmc")
    tflmcSrcDir = context.environment.paths["deps"].path / "src" / tflmcName
    if rebuild or not utils.is_populated(tflmcSrcDir):
        tflmcRepo = context.environment.repos["tflite_micro_compiler"]
        utils.clone(tflmcRepo.url, tflmcSrcDir, branch=tflmcRepo.ref)
    context.cache["tflmc.src_dir"] = tflmcSrcDir


def _validate_build_tflite_micro_compiler(context: MlonMcuContext, params=None):
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
def build_tflite_micro_compiler(context: MlonMcuContext, params=None, rebuild=False):
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
    if rebuild or not utils.is_populated(tflmcBuildDir) or not tflmcExe.is_file():
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
        utils.mkdirs(tflmcInstallDir)
        utils.move(tflmcBuildDir / "compiler", tflmcExe)
    context.cache["tflmc.build_dir", flags] = tflmcBuildDir
    context.cache["tflmc.exe", flags] = tflmcExe


#############
# riscv_gcc #
#############


def _validate_riscv_gcc(context: MlonMcuContext, params=None):
    if "etiss" not in enabled_targets:
        return False
    if params:
        if "vext" in params:
            if params["vext"]:
                if "vext" not in supported_features:
                    return False
    return True


@Tasks.provides(["riscv_gcc.install_dir"])
@Tasks.param("vext", [False, True])
@Tasks.validate(_validate_riscv_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(context: MlonMcuContext, params=None, rebuild=False):
    """Download and install the RISCV GCC toolchain."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["vext"], "vext"))
    riscvName = utils.makeDirName("riscv_gcc", flags=flags)
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    user_vars = context.environment.vars
    if "riscv_gcc.dir" in user_vars:
        # TODO: WARNING
        riscvInstallDir = user_vars["riscv_gcc.dir"]
    else:
        vext = False
        if "vext" in params:
            vext = params["vext"]
        riscvVersion = (
            user_vars["riscv.version"]
            if "riscv.version" in user_vars
            else ("8.3.0-2020.04.0" if not vext else "10.2.0-2020.12.8")
        )
        riscvDist = (
            user_vars["riscv.distribution"]
            if "riscv.distribution" in user_vars
            else "x86_64-linux-ubuntu14"
        )
        if vext:
            subdir = "v" + ".".join(riscvVersion.split("-")[1].split(".")[:-1])
            riscvUrl = (
                "https://static.dev.sifive.com/dev-tools/freedom-tools/" + subdir + "/"
            )
            riscvFileName = f"riscv64-unknown-elf-toolchain-{riscvVersion}-{riscvDist}"
        else:
            riscvUrl = "https://static.dev.sifive.com/dev-tools/"
            riscvFileName = f"riscv64-unknown-elf-gcc-{riscvVersion}-{riscvDist}"
        riscvArchive = riscvFileName + ".tar.gz"
        if rebuild or not utils.is_populated(riscvInstallDir):
            utils.download_and_extract(riscvUrl, riscvArchive, riscvInstallDir)
    context.cache["riscv_gcc.install_dir", flags] = riscvInstallDir


########
# llvm #
########


def _validate_llvm(context: MlonMcuContext, params=None):
    if "tvm" not in enabled_frameworks:
        return False
    return True


@Tasks.provides(["llvm.install_dir"])
@Tasks.validate(_validate_llvm)
@Tasks.register(category=TaskType.MISC)
def install_llvm(context: MlonMcuContext, params=None, rebuild=False):
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
    if rebuild or not utils.is_populated(llvmInstallDir):
        utils.download_and_extract(llvmUrl, llvmArchive, llvmInstallDir)
    context.cache["llvm.install_dir"] = llvmInstallDir


#########
# etiss #
#########


def _validate_etiss(context: MlonMcuContext, params={}):
    if "etiss" not in enabled_targets:
        return False
    return True


@Tasks.provides(["etiss.src_dir"])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def clone_etiss(context: MlonMcuContext, params=None, rebuild=False):
    """Clone the ETISS repository."""
    etissName = utils.makeDirName("etiss")
    etissSrcDir = context.environment.paths["deps"].path / "src" / etissName
    if rebuild or not utils.is_populated(etissSrcDir):
        etissRepo = context.environment.repos["etiss"]
        utils.clone(etissRepo.url, etissSrcDir, branch=etissRepo.ref)
    context.cache["etiss.src_dir"] = etissSrcDir


@Tasks.needs(["etiss.src_dir"])
@Tasks.provides(["etiss.build_dir", "etiss.install_dir"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etiss(context: MlonMcuContext, params=None, rebuild=False):
    """Build the ETISS simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.environment.paths["deps"].path / "build" / etissName
    etissInstallDir = context.environment.paths["deps"].path / "install" / etissName
    if rebuild or not utils.is_populated(etissBuildDir):
        utils.mkdirs(etissBuildDir)
        utils.cmake(
            context.cache["etiss.src_dir"],
            "-DCMAKE_INSTALL_PREFIX=" + str(etissInstallDir),
            cwd=etissBuildDir,
            debug=params["dbg"],
        )
        utils.make(cwd=etissBuildDir)
    context.cache["etiss.install_dir", flags] = etissInstallDir
    context.cache["etiss.build_dir", flags] = etissBuildDir


@Tasks.needs(["etiss.src_dir"])
@Tasks.provides(["etissvp.src_dir", "etiss.lib_dir"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def install_etiss(context: MlonMcuContext, params=None, rebuild=False):
    """Install ETISS."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.cache["etiss.build_dir", flags]
    etissInstallDir = context.cache["etiss.install_dir", flags]
    etissvpSrcDir = etissInstallDir / "examples" / "bare_etiss_processor"
    etissLibDir = etissInstallDir / "lib"
    if (
        rebuild
        or not utils.is_populated(etissvpSrcDir)
        or not utils.is_populated(etissLibDir)
    ):
        utils.make("install", cwd=etissBuildDir)
    context.cache["etissvp.src_dir", flags] = etissvpSrcDir
    context.cache["etiss.lib_dir", flags] = etissLibDir
    # return True


@Tasks.needs(["etissvp.src_dir"])
@Tasks.provides(["etissvp.build_dir", "etissvp.exe"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etissvp(context: MlonMcuContext, params=None, rebuild=False):
    """Build the ETISS virtual prototype."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissvpName = utils.makeDirName("build", flags=flags)
    etissvpSrcDir = context.cache["etissvp.src_dir", flags]
    etissvpBuildDir = etissvpSrcDir / etissvpName

    def addMemArgs(args, context=None):  # TODO: find out if this is still required?
        memMap = (0x0, 0x800000, 0x800000, 0x4000000)

        if context:
            user_vars = context.environment.vars
            if "etissvp.rom_start" in user_vars:
                temp = user_vars["etissvp.rom_start"]
                if not isinstance(temp, int):
                    temp = int(
                        temp, 0
                    )  # This should automatically detect the base via the prefix
                memMap[0] = temp
            if "etissvp.rom_size" in user_vars:
                temp = user_vars["etissvp.rom_size"]
                if not isinstance(temp, int):
                    temp = int(
                        temp, 0
                    )  # This should automatically detect the base via the prefix
                memMap[1] = temp
            if "etissvp.ram_start" in user_vars:
                temp = user_vars["etissvp.ram_start"]
                if not isinstance(temp, int):
                    temp = int(
                        temp, 0
                    )  # This should automatically detect the base via the prefix
                memMap[2] = temp
            if "etissvp.ram_size" in user_vars:
                temp = user_vars["etissvp.ram_size"]
                if not isinstance(temp, int):
                    temp = int(
                        temp, 0
                    )  # This should automatically detect the base via the prefix
                memMap[3] = temp

        def checkMemMap(mem):
            rom_start, rom_size, ram_start, ram_size = mem[0], mem[1], mem[2], mem[3]
            for val in mem:
                assert isinstance(val, int)

                def is_power_of_two(n):
                    return n == 0 or (n & (n - 1) == 0)

                assert is_power_of_two(val)
            assert rom_start + rom_size <= ram_start

        checkMemMap(memMap)

        args.append(f"-DPULPINO_ROM_START={hex(memMap[0])}")
        args.append(f"-DPULPINO_RAM_SIZE={hex(memMap[1])}")
        args.append(f"-DPULPINO_ROM_START={hex(memMap[2])}")
        args.append(f"-DPULPINO_RAM_SIZE={hex(memMap[3])}")
        return args

    etissvpArgs = addMemArgs([], context=context)
    utils.mkdirs(etissvpBuildDir)
    utils.cmake(etissvpSrcDir, *etissvpArgs, cwd=etissvpBuildDir, debug=params["dbg"])
    utils.make(cwd=etissvpBuildDir)
    context.cache["etissvp.build_dir", flags] = etissvpBuildDir
    etissvpExe = etissvpBuildDir / "bare_etiss_processor"
    context.cache["etissvp.exe", flags] = etissvpExe


#######
# tvm #
#######


def _validate_tvm(context: MlonMcuContext, params=None):
    if "tvm" not in supported_frameworks:
        return False
    return True


@Tasks.provides(["tvm.src_dir"])
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tvm(context: MlonMcuContext, params=None, rebuild=False):
    """Clone the TVM repository."""
    tvmName = utils.makeDirName("tvm")
    tvmSrcDir = context.environment.paths["deps"].path / "install" / tvmName
    if rebuild or not utils.is_populated(tvmSrcDir):
        tvmRepo = context.environment.repos["tvm"]
        utils.clone(tvmRepo.url, tvmSrcDir, branch=tvmRepo.ref, recursive=True)
    context.cache["tvm.src_dir"] = tvmSrcDir


@Tasks.needs(["tvm.src_dir", "llvm.install_dir"])
@Tasks.provides(["tvm.build_dir", "tvm.lib", "tvm.pythonpath"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tvm(context: MlonMcuContext, params=None, rebuild=False):
    """Build the TVM framework."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # FIXME: Try to use TVM dir outside of src dir to allow multiple versions/dbg etc!
    # This should help: TVM_LIBRARY_PATH -> tvm.build_dir
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.cache["tvm.src_dir"]
    tvmBuildDir = tvmSrcDir / "build"
    tvmLib = tvmBuildDir / "libtvm.so"
    tvmPythonPath = tvmSrcDir / "python"
    if rebuild or not utils.is_populated(tvmBuildDir) or not tvmLib.is_file():
        ninja = False
        if context:
            user_vars = context.environment.vars
            if "tvm.make_tool" in user_vars:
                if user_vars["tvm.make_tool"] == "ninja":
                    ninja = True
        utils.mkdirs(tvmBuildDir)
        cfgFileSrc = tvmSrcDir / "cmake" / "config.cmake"
        cfgFile = tvmBuildDir / "config.cmake"
        llvmConfig = str(context.cache["llvm.install_dir"] / "bin" / "llvm-config")
        llvmConfigEscaped = str(llvmConfig).replace("/", "\\/")
        utils.copy(cfgFileSrc, cfgFile)
        utils.exec(
            "sed",
            "-i",
            "--",
            "s/USE_LLVM \(OFF\|ON\)/USE_LLVM " + llvmConfigEscaped + "/g",
            cfgFile,
        )
        utils.cmake(tvmSrcDir, cwd=tvmBuildDir, debug=True, use_ninja=ninja)
        utils.make(cwd=tvmBuildDir, use_ninja=ninja)
    context.cache["tvm.build_dir", flags] = tvmBuildDir
    context.cache["tvm.lib", flags] = tvmLib
    context.cache["tvm.pythonpath"] = tvmPythonPath


##########
# utvmcg #
##########


def _validate_utvmcg(context: MlonMcuContext, params=None):
    if not _validate_tvm(context, params=params):
        return False
    if "tvmcg" not in supported_backends:
        return False
    return True


@Tasks.provides(["utvmcg.src_dir"])
@Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(context: MlonMcuContext, params=None, rebuild=False):
    """Clone the uTVM code generator."""
    utvmcgName = utils.makeDirName("utvmcg")
    utvmcgSrcDir = context.environment.paths["deps"].path / "src" / utvmcgName
    if rebuild or not utils.is_populated(utvmcgSrcDir):
        utvmcgRepo = context.environment.repos["utvm_staticrt_codegen"]
        utils.clone(utvmcgRepo.url, utvmcgSrcDir, branch=utvmcgRepo.ref)
    context.cache["utvmcg.src_dir"] = utvmcgSrcDir


@Tasks.needs(["utvmcg.src_dir", "tvm.src_dir"])
@Tasks.provides(["utvmcg.build_dir", "utvmcg.exe"])
@Tasks.param("dbg", False)
# @Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(context: MlonMcuContext, params=None, rebuild=False):
    """Build the uTVM code generator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    utvmcgName = utils.makeDirName("utvmcg", flags=flags)
    utvmcgSrcDir = context.cache["utvmcg.src_dir"]
    utvmcgBuildDir = context.environment.paths["deps"].path / "build" / utvmcgName
    utvmcgInstallDir = context.environment.paths["deps"].path / "install" / utvmcgName
    utvmcgExe = utvmcgInstallDir / "compiler"
    if rebuild or not utils.is_populated(utvmcgSrcDir) or not utvmcgExe.is_file():
        utvmcgArgs = []
        utvmcgArgs.append("-DTVM_DIR=" + str(context.cache["tvm.src_dir"]))
        crtConfigPath = (
            context.cache["tvm.src_dir"] / "apps" / "bundle_deploy" / "crt_config"
        )
        if context:
            user_vars = context.environment.vars
            if "tvm.crt_config_dir" in user_vars:
                crtConfigPath = Path(user_vars["tvm.crt_config_dir"])
        utvmcgArgs.append("-DTVM_CRT_CONFIG_DIR=" + str(crtConfigPath))
        utils.mkdirs(utvmcgBuildDir)
        utils.cmake(utvmcgSrcDir, *utvmcgArgs, cwd=utvmcgBuildDir, debug=params["dbg"])
        utils.make(cwd=utvmcgBuildDir)
        utils.mkdirs(utvmcgInstallDir)
        utils.move(utvmcgBuildDir / "utvm_staticrt_codegen", utvmcgExe)
    context.cache["utvmcg.build_dir", flags] = utvmcgBuildDir
    context.cache["utvmcg.exe", flags] = utvmcgExe


#############
# muriscvnn #
#############


def _validate_muriscvnn(context: MlonMcuContext, params=None):
    if not context.environment.supports_feature("muriscvnn"):
        return False
    assert "muriscvnn" in context.environment.repos, "Undefined repository: 'muriscvnn'"
    if params:
        if "vext" in params:
            pass
    return True


@Tasks.provides(["muriscvnn.src_dir"])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def clone_muriscvnn(context: MlonMcuContext, params=None, rebuild=False):
    """Clone the muRISCV-NN project."""
    muriscvnnName = utils.makeDirName("muriscvnn")
    muriscvnnSrcDir = context.environment.paths["deps"].path / "src" / muriscvnnName
    if rebuild or not utils.is_populated(muriscvnnSrcDir):
        muriscvnnRepo = context.environment.repos["muriscvnn"]
        utils.clone(muriscvnnRepo.url, muriscvnnSrcDir, branch=muriscvnnRepo.ref)
    context.cache["muriscvnn.src_dir"] = muriscvnnSrcDir


@Tasks.needs(["muriscvnn.src_dir", "etiss.install_dir", "riscv_gcc.install_dir"])
@Tasks.provides(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Tasks.param("dbg", [False, True])
@Tasks.param("vext", [False, True])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def build_muriscvnn(context: MlonMcuContext, params=None, rebuild=False):
    """Build muRISCV-NN."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"), (params["vext"], "vext"))
    flags_ = utils.makeFlags((params["vext"], "vext"))
    muriscvnnName = utils.makeDirName("muriscvnn", flags=flags)
    muriscvnnSrcDir = context.cache["muriscvnn.src_dir"]
    muriscvnnBuildDir = context.environment.paths["deps"].path / "build" / muriscvnnName
    muriscvnnIncludeDir = muriscvnnBuildDir / "Includes"
    if rebuild or not utils.is_populated(muriscvnnBuildDir):
        utils.mkdirs(muriscvnnBuildDir)
        muriscvnnArgs = []
        muriscvnnArgs.append(
            "-DRISCV_GCC_PREFIX=" + str(context.cache["riscv_gcc.install_dir", flags_])
        )
        vext = False
        if "vext" in params:
            vext = params["vext"]
        muriscvnnArgs.append("-DUSE_VEXT=" + ("ON" if vext else "OFF"))
        utils.cmake(
            muriscvnnSrcDir, *muriscvnnArgs, cwd=muriscvnnBuildDir, debug=params["dbg"]
        )
        utils.make(cwd=muriscvnnBuildDir)
    context.cache["muriscvnn.build_dir", flags] = muriscvnnBuildDir
    context.cache["muriscvnn.inc_dir", flags] = muriscvnnIncludeDir


# TODO: return True if changed, False if not, or: reurn code: unchanged, changed, error, unknown

# install_dependencies()
