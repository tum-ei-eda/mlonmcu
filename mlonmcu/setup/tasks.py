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
from pathlib import Path

from mlonmcu.setup.task import TaskFactory, TaskType
from mlonmcu.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

logger = get_logger()

Tasks = TaskFactory()


######
# tf #
######


def _validate_tensorflow(context: MlonMcuContext, params=None):
    return context.environment.has_framework("tflite")


@Tasks.provides(["tf.src_dir"])
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the TF/TFLM repository."""
    tfName = utils.makeDirName("tf")
    tfSrcDir = context.environment.paths["deps"].path / "src" / tfName
    if rebuild or not utils.is_populated(tfSrcDir):
        tfRepo = context.environment.repos["tensorflow"]
        utils.clone(tfRepo.url, tfSrcDir, branch=tfRepo.ref, refresh=rebuild)
    context.cache["tf.src_dir"] = tfSrcDir


@Tasks.needs(["tf.src_dir"])
@Tasks.provides(["tf.dl_dir", "tf.lib_path"])
# @Tasks.param("dbg", False)
@Tasks.param("dbg", True)
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tensorflow(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Download tensorflow dependencies and build lib."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # tfName = utils.makeDirName("tf", flags=flags)
    tfSrcDir = context.cache["tf.src_dir"]
    tflmDir = tfSrcDir / "tensorflow" / "lite" / "micro"
    tflmBuildDir = tflmDir / "tools" / "make"
    tflmDownloadsDir = tflmBuildDir / "downloads"
    if params["dbg"]:
        tflmLib = (
            tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
        )  # FIXME: add _dbg suffix is possible
    else:
        tflmLib = tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
    # if rebuild or not tflmLib.is_file() or not utils.is_populated(tflmDownloadsDir):
    if rebuild or not utils.is_populated(tflmDownloadsDir):
        tfDbgArg = ["BUILD_TYPE=debug"] if params["dbg"] else []
        utils.make(
            "-f",
            str(tflmDir / "tools" / "make" / "Makefile"),
            "third_party_downloads",
            *tfDbgArg,
            cwd=tfSrcDir,
            live=verbose,
        )
    context.cache["tf.dl_dir"] = tflmDownloadsDir
    context.cache["tf.lib_path", flags] = tflmLib  # ignore!


#########
# tflmc #
#########


def _validate_tflite_micro_compiler(context: MlonMcuContext, params=None):
    if not _validate_tensorflow(context, params=params):
        return False
    if not context.environment.has_backend("tflmc"):
        return False
    return True


@Tasks.provides(["tflmc.src_dir"])
@Tasks.validate(_validate_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
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
                if not context.environment.supports_feature("muriscvnn"):
                    return False
    return _validate_tflite_micro_compiler(context, params=params)


@Tasks.needs(["tflmc.src_dir", "tf.src_dir"])
@Tasks.optional(["muriscvnn.build_dir", "muriscvnn.inc_dir"])
@Tasks.provides(["tflmc.build_dir", "tflmc.exe"])
@Tasks.param("muriscvnn", [False, True])
@Tasks.param("dbg", False)
@Tasks.param("arch", ["x86"])  # TODO: compile for arm/riscv in the future
@Tasks.validate(_validate_build_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
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
            live=verbose,
        )
        utils.make(cwd=tflmcBuildDir, live=verbose)
        utils.mkdirs(tflmcInstallDir)
        utils.move(tflmcBuildDir / "compiler", tflmcExe)
    context.cache["tflmc.build_dir", flags] = tflmcBuildDir
    context.cache["tflmc.exe", flags] = tflmcExe


#############
# riscv_gcc #
#############


def _validate_riscv_gcc(context: MlonMcuContext, params=None):
    if not (
        context.environment.has_target("etiss_pulpino")
        or context.environment.has_target("spike")
        or context.environment.has_target("ovpsim")
    ):
        return False
    if params:
        if "vext" in params:
            if params["vext"]:
                if not context.environment.has_feature("vext"):
                    return False
    return True


@Tasks.provides(["riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.param("vext", [False, True])
@Tasks.validate(_validate_riscv_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Download and install the RISCV GCC toolchain."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["vext"], "vext"))
    riscvName = utils.makeDirName("riscv_gcc", flags=flags)
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    user_vars = context.environment.vars
    if "riscv_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # TODO: WARNING
        riscvInstallDir = user_vars["riscv_gcc.install_dir"]
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
    else:
        vext = False
        if "vext" in params:
            vext = params["vext"]
        if "riscv_gcc.dl_url" in user_vars:
            fullUrlSplit = user_vars["riscv_gcc.dl_url"].split("/")
            riscvUrl = "/".join(fullUrlSplit[:-1])
            riscvFileName, riscvFileExtension = fullUrlSplit[-1].split(".", 1)
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
            utils.download_and_extract(riscvUrl, riscvArchive, riscvInstallDir)
    gccNames = ["riscv64-unknown-elf", "riscv32-unknown-elf"]
    gccName = None
    for name in gccNames:
        if (riscvInstallDir / name).is_dir():
            gccName = name
            break
    assert gccName is not None, "Toolchain name could not be dtemined automatically"
    context.cache["riscv_gcc.install_dir", flags] = riscvInstallDir
    context.cache["riscv_gcc.name", flags] = gccName


########
# llvm #
########


def _validate_llvm(context: MlonMcuContext, params=None):
    return context.environment.has_framework("tvm")


@Tasks.provides(["llvm.install_dir"])
@Tasks.validate(_validate_llvm)
@Tasks.register(category=TaskType.MISC)
def install_llvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Download and install LLVM."""
    llvmName = utils.makeDirName("llvm")
    llvmInstallDir = context.environment.paths["deps"].path / "install" / llvmName
    user_vars = context.environment.vars
    if "llvm.install_dir" in user_vars:  # TODO: also check command line flags?
        # TODO: WARNING
        llvmInstallDir = user_vars["llvm.install_dir"]
    else:
        llvmVersion = user_vars["llvm.version"] if "llvm.version" in user_vars else "11.0.1"
        llvmDist = (
            user_vars["llvm.distribution"] if "llvm.distribution" in user_vars else "x86_64-linux-gnu-ubuntu-16.04"
        )
        llvmUrl = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvmVersion}/"
        llvmFileName = f"clang+llvm-{llvmVersion}-{llvmDist}"
        llvmArchive = llvmFileName + ".tar.xz"
        # if rebuild or not utils.is_populated(llvmInstallDir):
        # rebuild should only be triggered if the version/url changes but we can not detect that at the moment
        if not utils.is_populated(llvmInstallDir):
            utils.download_and_extract(llvmUrl, llvmArchive, llvmInstallDir)
    context.cache["llvm.install_dir"] = llvmInstallDir


#########
# etiss #
#########


def _validate_etiss(context: MlonMcuContext, params={}):
    return context.environment.has_target("etiss_pulpino")


@Tasks.provides(["etiss.src_dir"])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def clone_etiss(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the ETISS repository."""
    etissName = utils.makeDirName("etiss")
    etissSrcDir = context.environment.paths["deps"].path / "src" / etissName
    if rebuild or not utils.is_populated(etissSrcDir):
        etissRepo = context.environment.repos["etiss"]
        utils.clone(etissRepo.url, etissSrcDir, branch=etissRepo.ref)
    context.cache["etiss.src_dir"] = etissSrcDir


@Tasks.needs(["etiss.src_dir", "llvm.install_dir"])
@Tasks.provides(["etiss.build_dir", "etiss.install_dir"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etiss(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Build the ETISS simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.environment.paths["deps"].path / "build" / etissName
    etissInstallDir = context.environment.paths["deps"].path / "install" / etissName
    llvmInstallDir = context.cache["llvm.install_dir"]
    if rebuild or not utils.is_populated(etissBuildDir):
        utils.mkdirs(etissBuildDir)
        env = os.environ.copy()
        # env["LLVM_DIR"] = str(llvmInstallDir)
        utils.cmake(
            context.cache["etiss.src_dir"],
            "-DCMAKE_INSTALL_PREFIX=" + str(etissInstallDir),
            cwd=etissBuildDir,
            debug=params["dbg"],
            env=env,
            live=verbose,
        )
        utils.make(cwd=etissBuildDir, live=verbose)
    context.cache["etiss.install_dir", flags] = etissInstallDir
    context.cache["etiss.build_dir", flags] = etissBuildDir


@Tasks.needs(["etiss.build_dir"])
@Tasks.provides(["etissvp.src_dir", "etiss.lib_dir", "etiss.install_dir"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def install_etiss(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Install ETISS."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.cache["etiss.build_dir", flags]
    etissInstallDir = context.cache["etiss.install_dir", flags]
    etissvpSrcDir = etissInstallDir / "examples" / "bare_etiss_processor"
    etissLibDir = etissInstallDir / "lib"
    if rebuild or not utils.is_populated(etissvpSrcDir) or not utils.is_populated(etissLibDir):
        utils.make("install", cwd=etissBuildDir, live=verbose)
    context.cache["etissvp.src_dir", flags] = etissvpSrcDir
    context.cache["etiss.lib_dir", flags] = etissLibDir
    context.cache["etiss.install_dir", flags] = etissInstallDir
    # return True


@Tasks.needs(["etissvp.src_dir"])
@Tasks.provides(["etissvp.build_dir", "etissvp.exe"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etissvp(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Build the ETISS virtual prototype."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissvpName = utils.makeDirName("build", flags=flags)
    etissvpSrcDir = context.cache["etissvp.src_dir", flags]
    etissvpBuildDir = etissvpSrcDir / etissvpName
    etissvpExe = etissvpBuildDir / "main"

    if rebuild or not etissvpExe.is_file():

        def addMemArgs(args, context=None):  # TODO: find out if this is still required?
            memMap = (0x0, 0x800000, 0x800000, 0x4000000)

            if context:
                user_vars = context.environment.vars
                if "etissvp.rom_start" in user_vars:
                    temp = user_vars["etissvp.rom_start"]
                    if not isinstance(temp, int):
                        temp = int(temp, 0)  # This should automatically detect the base via the prefix
                    memMap[0] = temp
                if "etissvp.rom_size" in user_vars:
                    temp = user_vars["etissvp.rom_size"]
                    if not isinstance(temp, int):
                        temp = int(temp, 0)  # This should automatically detect the base via the prefix
                    memMap[1] = temp
                if "etissvp.ram_start" in user_vars:
                    temp = user_vars["etissvp.ram_start"]
                    if not isinstance(temp, int):
                        temp = int(temp, 0)  # This should automatically detect the base via the prefix
                    memMap[2] = temp
                if "etissvp.ram_size" in user_vars:
                    temp = user_vars["etissvp.ram_size"]
                    if not isinstance(temp, int):
                        temp = int(temp, 0)  # This should automatically detect the base via the prefix
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
        utils.cmake(
            etissvpSrcDir,
            *etissvpArgs,
            cwd=etissvpBuildDir,
            debug=params["dbg"],
            live=verbose,
        )
        utils.make(cwd=etissvpBuildDir, live=verbose)
    context.cache["etissvp.build_dir", flags] = etissvpBuildDir
    context.cache["etissvp.exe", flags] = etissvpExe


#######
# tvm #
#######


def _validate_tvm(context: MlonMcuContext, params=None):
    if "patch" in params and bool(params["patch"]):
        if not context.environment.has_feature("disable_legalize"):
            return False

    return context.environment.has_framework("tvm")


@Tasks.provides(["tvm.src_dir"])
@Tasks.optional(["tvm_extensions.src_dir"])
@Tasks.validate(_validate_tvm)
@Tasks.param("patch", [False, True])  # This is just a temporary workaround until the patch is hopefully upstreamed
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the TVM repository."""
    if not params:
        params = {}
    patch = params["patch"]
    flags = utils.makeFlags((patch, "patch"))
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.environment.paths["deps"].path / "src" / tvmName
    tvmPythonPath = tvmSrcDir / "python"
    if rebuild or not utils.is_populated(tvmSrcDir):
        tvmRepo = context.environment.repos["tvm"]
        utils.clone(tvmRepo.url, tvmSrcDir, branch=tvmRepo.ref, recursive=True)
        if patch:
            extSrcDir = context.cache["tvm_extensions.src_dir"]
            patchFile = extSrcDir / "tvmc_diff.patch"
            utils.apply(tvmSrcDir, patchFile)

    context.cache["tvm.src_dir", flags] = tvmSrcDir
    context.cache["tvm.pythonpath", flags] = tvmPythonPath


@Tasks.needs(["tvm.src_dir", "llvm.install_dir"])
@Tasks.provides(["tvm.build_dir", "tvm.lib"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Build the TVM framework."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    dbg = bool(params["dbg"])
    # FIXME: Try to use TVM dir outside of src dir to allow multiple versions/dbg etc!
    # This should help: TVM_LIBRARY_PATH -> tvm.build_dir
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.cache["tvm.src_dir", ()]  # params["patch"] does not affect the build
    tvmBuildDir = context.environment.paths["deps"].path / "build" / tvmName
    tvmLib = tvmBuildDir / "libtvm.so"
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
            r"s/USE_MICRO OFF/USE_MICRO ON/g",
            str(cfgFile),
        )
        utils.exec(
            "sed",
            "-i",
            "--",
            r"s/USE_MICRO_STANDALONE_RUNTIME OFF/USE_MICRO_STANDALONE_RUNTIME ON/g",
            str(cfgFile),
        )
        utils.exec(
            "sed",
            "-i",
            "--",
            r"s/USE_LLVM \(OFF\|ON\)/USE_LLVM " + llvmConfigEscaped + "/g",
            str(cfgFile),
        )
        utils.cmake(tvmSrcDir, cwd=tvmBuildDir, debug=dbg, use_ninja=ninja, live=verbose)
        utils.make(cwd=tvmBuildDir, use_ninja=ninja, live=verbose)
    context.cache["tvm.build_dir", flags] = tvmBuildDir
    context.cache["tvm.lib", flags] = tvmLib


##########
# utvmcg #
##########


def _validate_utvmcg(context: MlonMcuContext, params=None):
    if not _validate_tvm(context, params=params):
        return False
    return context.environment.has_backend("tvmcg")


@Tasks.provides(["utvmcg.src_dir"])
@Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
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
@Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
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
        crtConfigPath = context.cache["tvm.src_dir"] / "apps" / "bundle_deploy" / "crt_config"
        if context:
            user_vars = context.environment.vars
            if "tvm.crt_config_dir" in user_vars:
                crtConfigPath = Path(user_vars["tvm.crt_config_dir"])
        utvmcgArgs.append("-DTVM_CRT_CONFIG_DIR=" + str(crtConfigPath))
        utils.mkdirs(utvmcgBuildDir)
        utils.cmake(
            utvmcgSrcDir,
            *utvmcgArgs,
            cwd=utvmcgBuildDir,
            debug=params["dbg"],
            live=verbose,
        )
        utils.make(cwd=utvmcgBuildDir, live=verbose)
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


@Tasks.provides(["muriscvnn.src_dir", "muriscvnn.inc_dir"])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def clone_muriscvnn(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the muRISCV-NN project."""
    muriscvnnName = utils.makeDirName("muriscvnn")
    muriscvnnSrcDir = context.environment.paths["deps"].path / "src" / muriscvnnName
    muriscvnnIncludeDir = muriscvnnSrcDir / "Include"
    user_vars = context.environment.vars
    if "muriscvnn.lib" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(muriscvnnSrcDir):
        muriscvnnRepo = context.environment.repos["muriscvnn"]
        utils.clone(muriscvnnRepo.url, muriscvnnSrcDir, branch=muriscvnnRepo.ref)
    context.cache["muriscvnn.src_dir"] = muriscvnnSrcDir
    context.cache["muriscvnn.inc_dir"] = muriscvnnIncludeDir


@Tasks.needs(["muriscvnn.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
# @Tasks.optional(["riscv_gcc.install_dir", "riscv_gcc.name", "arm_gcc.install_dir"])
@Tasks.provides(["muriscvnn.build_dir", "muriscvnn.lib"])
@Tasks.param("dbg", [False, True])
@Tasks.param("vext", [False, True])
# @Tasks.param("target_arch", ["x86", "riscv", "arm"])  # TODO: implement
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def build_muriscvnn(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Build muRISCV-NN."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"), (params["vext"], "vext"))
    flags_ = utils.makeFlags((params["vext"], "vext"))
    muriscvnnName = utils.makeDirName("muriscvnn", flags=flags)
    muriscvnnSrcDir = context.cache["muriscvnn.src_dir"]
    muriscvnnBuildDir = context.environment.paths["deps"].path / "build" / muriscvnnName
    muriscvnnInstallDir = context.environment.paths["deps"].path / "install" / muriscvnnName
    muriscvnnLib = muriscvnnInstallDir / "libmuriscvnn.a"
    user_vars = context.environment.vars
    if "muriscvnn.lib" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(muriscvnnBuildDir) and muriscvnnLib.is_file()):
        utils.mkdirs(muriscvnnBuildDir)
        gccName = context.cache["riscv_gcc.name", flags_]
        assert gccName == "riscv32-unknown-elf", "muRISCV-NN requires a non-multilib toolchain!"
        muriscvnnArgs = []
        muriscvnnArgs.append("-DRISCV_GCC_PREFIX=" + str(context.cache["riscv_gcc.install_dir", flags_]))
        vext = False
        if "vext" in params:
            vext = params["vext"]
        muriscvnnArgs.append("-DUSE_VEXT=" + ("ON" if vext else "OFF"))
        muriscvnnArgs.append(f"-DRISCV_GCC_BASENAME={gccName}")
        utils.cmake(
            muriscvnnSrcDir,
            *muriscvnnArgs,
            cwd=muriscvnnBuildDir,
            debug=params["dbg"],
            live=verbose,
        )
        utils.make(cwd=muriscvnnBuildDir, live=verbose)
        utils.mkdirs(muriscvnnInstallDir)
        utils.move(muriscvnnBuildDir / "Source" / "libmuriscv_nn.a", muriscvnnLib)
    context.cache["muriscvnn.build_dir", flags] = muriscvnnBuildDir
    context.cache["muriscvnn.lib", flags] = muriscvnnLib


def _validate_spike(context: MlonMcuContext, params=None):
    if not context.environment.has_target("spike"):
        return False
    assert "spikepk" in context.environment.repos, "Undefined repository: 'spikepk'"
    assert "spike" in context.environment.repos, "Undefined repository: 'spike'"
    return True


@Tasks.provides(["spikepk.src_dir"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_spike_pk(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the spike proxt kernel."""
    spikepkName = utils.makeDirName("spikepk")
    spikepkSrcDir = context.environment.paths["deps"].path / "src" / spikepkName
    user_vars = context.environment.vars
    if "spike.pk" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(spikepkSrcDir):
        spikepkRepo = context.environment.repos["spikepk"]
        utils.clone(spikepkRepo.url, spikepkSrcDir, branch=spikepkRepo.ref)
    context.cache["spikepk.src_dir"] = spikepkSrcDir


@Tasks.needs(["spikepk.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.provides(["spikepk.build_dir", "spike.pk"])
# @Tasks.param("vext", [False, True])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def build_spike_pk(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Build Spike proxy kernel."""
    if not params:
        params = {}
    # flags = utils.makeFlags((params["vext"], "vext"))
    spikepkName = utils.makeDirName("spikepk")
    spikepkSrcDir = context.cache["spikepk.src_dir"]
    spikepkBuildDir = context.environment.paths["deps"].path / "build" / spikepkName
    spikepkInstallDir = context.environment.paths["deps"].path / "install" / spikepkName
    spikepkBin = spikepkInstallDir / "pk"
    user_vars = context.environment.vars
    if "spike.pk" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(spikepkBuildDir) and spikepkBin.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(spikepkBuildDir)
        gccName = context.cache["riscv_gcc.name"]
        assert gccName == "riscv32-unknown-elf", "Spike PK requires a non-multilib toolchain!"
        spikepkArgs = []
        spikepkArgs.append("--prefix=" + str(context.cache["riscv_gcc.install_dir"]))
        spikepkArgs.append("--host=" + gccName)
        spikepkArgs.append("--with-arch=rv32gcv")
        spikepkArgs.append("--with-abi=ilp32d")
        env = os.environ.copy()
        env["PATH"] = str(Path(context.cache["riscv_gcc.install_dir"]) / "bin") + ":" + env["PATH"]
        utils.exec_getout(
            str(spikepkSrcDir / "configure"),
            *spikepkArgs,
            cwd=spikepkBuildDir,
            env=env,
            live=verbose,
        )
        utils.make(cwd=spikepkBuildDir, live=verbose, env=env)
        utils.mkdirs(spikepkInstallDir)
        utils.move(spikepkBuildDir / "pk", spikepkBin)
    context.cache["spikepk.build_dir"] = spikepkBuildDir
    context.cache["spike.pk"] = spikepkBin


@Tasks.provides(["spike.src_dir"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_spike(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the spike simulator."""
    spikeName = utils.makeDirName("spike")
    spikeSrcDir = context.environment.paths["deps"].path / "src" / spikeName
    user_vars = context.environment.vars
    if "spike.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(spikeSrcDir):
        spikeRepo = context.environment.repos["spike"]
        utils.clone(spikeRepo.url, spikeSrcDir, branch=spikeRepo.ref)
    context.cache["spike.src_dir"] = spikeSrcDir


@Tasks.needs(["spike.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.provides(["spike.build_dir", "spike.exe"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def build_spike(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Build Spike simulator."""
    if not params:
        params = {}
    spikeName = utils.makeDirName("spike")
    spikeSrcDir = context.cache["spike.src_dir"]
    spikeBuildDir = context.environment.paths["deps"].path / "build" / spikeName
    spikeInstallDir = context.environment.paths["deps"].path / "install" / spikeName
    spikeExe = spikeInstallDir / "spike"
    user_vars = context.environment.vars
    if "spike.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(spikeBuildDir) and spikeExe.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(spikeBuildDir)
        spikeArgs = []
        spikeArgs.append("--prefix=" + str(context.cache["riscv_gcc.install_dir"]))
        utils.exec_getout(
            str(spikeSrcDir / "configure"),
            *spikeArgs,
            cwd=spikeBuildDir,
            live=verbose,
        )
        utils.make(cwd=spikeBuildDir, live=verbose)
        utils.mkdirs(spikeInstallDir)
        utils.move(spikeBuildDir / "spike", spikeExe)
    context.cache["spike.build_dir"] = spikeBuildDir
    context.cache["spike.exe"] = spikeExe


def _validate_cmsisnn(context: MlonMcuContext, params=None):
    return context.environment.has_feature("cmsisnn") or context.environment.has_feature("cmsisnnbyoc")


def _validate_cmsis(context: MlonMcuContext, params=None):
    return _validate_cmsisnn(context, params=params) or context.environment.has_target("corstone300")


@Tasks.provides(["cmsisnn.dir"])
@Tasks.validate(_validate_cmsis)
@Tasks.register(category=TaskType.MISC)
def clone_cmsis(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """CMSIS repository."""
    cmsisName = utils.makeDirName("cmsis")
    cmsisSrcDir = context.environment.paths["deps"].path / "src" / cmsisName
    # TODO: allow to skip this if cmsisnn.dir+cmsisnn.lib are provided by the user and corstone is not used -> move those checks to validate?
    if rebuild or not utils.is_populated(cmsisSrcDir):
        cmsisRepo = context.environment.repos["cmsis"]
        utils.clone(cmsisRepo.url, cmsisSrcDir, branch=cmsisRepo.ref, refresh=rebuild)
    context.cache["cmsisnn.dir"] = cmsisSrcDir


@Tasks.needs(["cmsisnn.dir"])
@Tasks.optional(["riscv_gcc.install_dir", "riscv_gcc.name", "arm_gcc.install_dir"])
@Tasks.provides(["cmsisnn.lib"])
@Tasks.param("dbg", [False, True])
@Tasks.param("target_arch", ["x86", "riscv", "arm"])
@Tasks.param("mvei", False)  # TODO: build?
@Tasks.param("dsp", False)  # TODO: build?
@Tasks.validate(_validate_cmsisnn)
@Tasks.register(category=TaskType.OPT)  # TODO: rename to TaskType.FEATURE?
def build_cmsisnn(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    flags = utils.makeFlags(
        (params["target_arch"], params["target_arch"]),
        (params["mvei"], "mvei"),
        (params["dsp"], "dsp"),
        (params["dbg"], "dbg"),
    )
    cmsisnnName = utils.makeDirName("cmsisnn", flags=flags)
    cmsisnnBuildDir = context.environment.paths["deps"].path / "build" / cmsisnnName
    cmsisnnInstallDir = context.environment.paths["deps"].path / "install" / cmsisnnName
    cmsisnnLib = cmsisnnInstallDir / "libcmsis-nn.a"
    cmsisSrcDir = context.cache["cmsisnn.dir"]
    cmsisnnSrcDir = cmsisSrcDir / "CMSIS" / "NN"
    if rebuild or not utils.is_populated(cmsisnnBuildDir) or not cmsisnnLib.is_file():
        utils.mkdirs(cmsisnnBuildDir)
        cmakeArgs = []
        env = os.environ.copy()
        # utils.cmake("-DTF_SRC=" + str(tfSrcDir), str(tflmcSrcDir), debug=params["dbg"], cwd=tflmcBuildDir)
        if params["target_arch"] == "arm":
            toolchainFile = cmsisSrcDir / "CMSIS" / "DSP" / "gcc.cmake"
            armCpu = "cortex-m55"  # TODO: make this variable?
            cmakeArgs.append(f"-DARM_CPU={armCpu}")
            cmakeArgs.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchainFile}")  # Why does this not set CMAKE_C_COMPILER?
            armBinDir = Path(context.cache["arm_gcc.install_dir"]) / "bin"
            cmakeArgs.append("-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY")
            old = env["PATH"]
            env["PATH"] = f"{armBinDir}:{old}"
        elif params["target_arch"] == "riscv":
            riscvPrefix = context.cache["riscv_gcc.install_dir"]
            riscvBasename = context.cache["riscv_gcc.name"]
            cmakeArgs.append(f"-DCMAKE_C_COMPILER={riscvPrefix}/bin/{riscvBasename}-gcc")
            # cmakeArgs.append("-DCMAKE_CXX_COMPILER={riscvprefix}/bin/{riscvBasename}-g++")
            # cmakeArgs.append("-DCMAKE_ASM_COMPILER={riscvprefix}/bin/{riscvBasename}-gcc")
            # cmakeArgs.append("-DCMAKE_EXE_LINKER_FLAGS=\"'-march=rv32gc' '-mabi=ilp32d'\"")  # TODO: How about vext?
            # cmakeArgs.append("-E env LDFLAGS=\"-march=rv32gc -mabi=ilp32d\"")
            # cmakeArgs.append("-E env LDFLAGS=\"-march=rv32gc -mabi=ilp32d\"")
            env["LDFLAGS"] = "-march=rv32gc -mabi=ilp32d"
            cmakeArgs.append("-DCMAKE_SYSTEM_NAME=Generic")
            # TODO: how about linker, objcopy, ar?
        elif params["target_arch"] == "x86":
            pass
        else:
            arch = params["target_arch"]
            raise ValueError(f"Target architecture '{arch}' is not supported")

        utils.cmake(
            *cmakeArgs,
            str(cmsisnnSrcDir),
            debug=params["dbg"],
            cwd=cmsisnnBuildDir,
            live=verbose,
            env=env,
        )
        utils.make(cwd=cmsisnnBuildDir, live=verbose)
        utils.mkdirs(cmsisnnInstallDir)
        utils.move(cmsisnnBuildDir / "Source" / "libcmsis-nn.a", cmsisnnLib)
    context.cache["cmsisnn.lib", flags] = cmsisnnLib


def _validate_corstone300(context: MlonMcuContext, params=None):
    return context.environment.has_target("corstone300")


@Tasks.provides(["arm_gcc.install_dir"])
@Tasks.validate(_validate_corstone300)
@Tasks.register(category=TaskType.TARGET)
def install_arm_gcc(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Download and install GNU compiler toolchain from ARM."""
    armName = utils.makeDirName("arm_gcc")
    armInstallDir = context.environment.paths["deps"].path / "install" / armName
    user_vars = context.environment.vars
    if "arm_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # armInstallDir = user_vars["riscv_gcc.install_dir"]
        return False
    else:
        if not utils.is_populated(armInstallDir):
            armUrl = "https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/"
            armFileName = "gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux"
            armArchive = armFileName + ".tar.bz2"
            utils.download_and_extract(armUrl, armArchive, armInstallDir)
    context.cache["arm_gcc.install_dir"] = armInstallDir


@Tasks.provides(["corstone300.exe"])
@Tasks.validate(_validate_corstone300)
@Tasks.register(category=TaskType.TARGET)
def install_corstone300(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Download and install corstone300 FVP from ARM."""
    fvpName = utils.makeDirName("corstone300")
    fvpInstallDir = context.environment.paths["deps"].path / "install" / fvpName
    fvpSubDir = fvpInstallDir / "fvp"
    fvpExe = fvpSubDir / "models" / "Linux64_GCC-6.4" / "FVP_Corstone_SSE-300_Ethos-U55"
    user_vars = context.environment.vars
    if "corstone300.exe" in user_vars:  # TODO: also check command line flags?
        # fvpExe = user_vars["corstone300.exe"]
        return False
    else:
        if not fvpExe.is_file():
            fvpUrl = "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/"
            fvpFileName = "FVP_Corstone_SSE-300_11.16_26"
            fvpArchive = fvpFileName + ".tgz"
            utils.download_and_extract(fvpUrl, fvpArchive, fvpInstallDir)
            fvpScript = fvpInstallDir / "FVP_Corstone_SSE-300.sh"
            utils.exec_getout(
                fvpScript, "--i-agree-to-the-contained-eula", "--no-interactive", "-d", fvpSubDir, print_output=False
            )
    context.cache["corstone300.exe"] = fvpExe


def _validate_tvm_extensions(context: MlonMcuContext, params=None):
    return _validate_tvm(context, params=params) and context.environment.has_feature("disable_legalize")


@Tasks.provides(["tvm_extensions.src_dir", "tvm_extensions.wrapper"])
@Tasks.validate(_validate_tvm_extensions)
@Tasks.register(category=TaskType.FEATURE)
def clone_tvm_extensions(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the TVM extensions repository."""
    extName = utils.makeDirName("tvm_extensions")
    extSrcDir = context.environment.paths["deps"].path / "src" / extName
    extWrapper = extSrcDir / "tvmc_wrapper.py"
    if rebuild or not utils.is_populated(extSrcDir):
        extRepo = context.environment.repos["tvm_extensions"]
        utils.clone(extRepo.url, extSrcDir, branch=extRepo.ref, refresh=rebuild)
    context.cache["tvm_extensions.src_dir"] = extSrcDir
    context.cache["tvm_extensions.wrapper"] = extWrapper

def _validate_mlif(context: MlonMcuContext, params=None):
    return context.environment.has_platform("mlif")


@Tasks.provides(["mlif.src_dir"])
@Tasks.validate(_validate_mlif)
@Tasks.register(category=TaskType.PLATFORM)
def clone_mlif(context: MlonMcuContext, params=None, rebuild=False, verbose=False):
    """Clone the MLonMCU SW repository."""
    mlifName = utils.makeDirName("mlif")
    mlifSrcDir = context.environment.paths["deps"].path / "src" / mlifName
    if rebuild or not utils.is_populated(mlifSrcDir):
        mlifRepo = context.environment.repos["mlif"]
        utils.clone(mlifRepo.url, mlifSrcDir, branch=mlifRepo.ref, refresh=rebuild)
    context.cache["mlif.src_dir"] = mlifSrcDir
