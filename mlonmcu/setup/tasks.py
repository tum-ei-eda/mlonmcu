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
    return context.environment.has_framework("tflm")


@Tasks.provides(["tf.src_dir"])
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
def build_tensorflow(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download tensorflow dependencies and build lib."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # tfName = utils.makeDirName("tf", flags=flags)
    tfSrcDir = context.cache["tf.src_dir"]
    tflmDir = Path(tfSrcDir) / "tensorflow" / "lite" / "micro"
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
            threads=threads,
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
def clone_tflite_micro_compiler(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the preinterpreter repository."""
    tflmcName = utils.makeDirName("tflmc")
    tflmcSrcDir = context.environment.paths["deps"].path / "src" / tflmcName
    if rebuild or not utils.is_populated(tflmcSrcDir):
        tflmcRepo = context.environment.repos["tflite_micro_compiler"]
        utils.clone(tflmcRepo.url, tflmcSrcDir, branch=tflmcRepo.ref)
    context.cache["tflmc.src_dir"] = tflmcSrcDir


def _validate_build_tflite_micro_compiler(context: MlonMcuContext, params=None):
    if params:
        muriscvnn = params.get("muriscvnn", False)
        cmsisnn = params.get("cmsisnn", False)
        if muriscvnn and cmsisnn:
            # Not allowed
            return False
        elif muriscvnn:
            if not context.environment.supports_feature("muriscvnn"):
                return False
        elif cmsisnn:
            if not context.environment.supports_feature("cmsisnn"):
                return False
    return _validate_tflite_micro_compiler(context, params=params)


@Tasks.needs(["tflmc.src_dir", "tf.src_dir"])
@Tasks.optional(["muriscvnn.lib", "muriscvnn.inc_dir", "cmsisnn.dir"])
@Tasks.provides(["tflmc.build_dir", "tflmc.exe"])
@Tasks.param("muriscvnn", [False, True])
@Tasks.param("cmsisnn", [False, True])
@Tasks.param("dbg", False)
@Tasks.param("arch", ["x86"])  # TODO: compile for arm/riscv in the future
@Tasks.validate(_validate_build_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the TFLM preinterpreter."""
    muriscvnn = params.get("muriscvnn", False)
    cmsisnn = params.get("cmsisnn", False)
    dbg = params.get("dbg", False)
    arch = params.get("arch", "x86")
    flags = utils.makeFlags((True, "arch"), (muriscvnn, "muriscvnn"), (cmsisnn, "cmsisnn"), (dbg, "dbg"))
    flags_ = utils.makeFlags((dbg, "dbg"))
    flags__ = utils.makeFlags((True, arch), (dbg, "dbg"))
    tflmcName = utils.makeDirName("tflmc", flags=flags)
    tflmcBuildDir = context.environment.paths["deps"].path / "build" / tflmcName
    tflmcInstallDir = context.environment.paths["deps"].path / "install" / tflmcName
    tflmcExe = tflmcInstallDir / "compiler"
    tfSrcDir = context.cache["tf.src_dir", flags_]
    tflmcSrcDir = context.cache["tflmc.src_dir", flags_]
    if rebuild or not utils.is_populated(tflmcBuildDir) or not tflmcExe.is_file():
        cmakeArgs = [
            "-DTF_SRC=" + str(tfSrcDir),
            "-DGET_TF_SRC=ON",
        ]
        if muriscvnn:
            muriscvnnLib = context.cache["muriscvnn.lib", flags__]
            muriscvnnInc = context.cache["muriscvnn.in_dir"]
            cmakeArgs.append("-DTFLM_OPTIMIZED_KERNEL=cmsis_nn")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_LIB={muriscvnnLib}")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_INCLUDE_DIR={muriscvnnInc}")
        elif cmsisnn:
            cmsisnnLib = context.cache["cmsisnn.lib", flags__]
            cmsisDir = Path(context.cache["cmsisnn.dir"])
            cmsisIncs = [
                str(cmsisDir),
                str(cmsisDir / "CMSIS" / "Core" / "Include"),
                str(cmsisDir / "CMSIS" / "NN" / "Include"),
                str(cmsisDir / "CMSIS" / "DSP" / "Include"),
            ]
            cmakeArgs.append("-DTFLM_OPTIMIZED_KERNEL=cmsis_nn")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_LIB={cmsisnnLib}")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_INCLUDE_DIR={cmsisIncs}")
        utils.mkdirs(tflmcBuildDir)
        # utils.cmake("-DTF_SRC=" + str(tfSrcDir), str(tflmcSrcDir), debug=params["dbg"], cwd=tflmcBuildDir)
        utils.cmake(
            *cmakeArgs,
            str(tflmcSrcDir),
            debug=dbg,
            cwd=tflmcBuildDir,
            live=verbose,
        )
        utils.make(cwd=tflmcBuildDir, threads=threads, live=verbose)
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
        vext = params.get("vext", False)
        pext = params.get("pext", False)
        if vext and pext:
            return False  # TODO: allow as soon as there is a compiler for this
        elif vext:
            if not context.environment.has_feature("vext"):
                return False
        elif pext:
            if not context.environment.has_feature("pext"):
                return False
    return True


@Tasks.provides(["riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.param("vext", [False, True])
@Tasks.param("pext", [False, True])
@Tasks.validate(_validate_riscv_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install the RISCV GCC toolchain."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["vext"], "vext"), (params["pext"], "pext"))
    riscvName = utils.makeDirName("riscv_gcc", flags=flags)
    riscvInstallDir = context.environment.paths["deps"].path / "install" / riscvName
    user_vars = context.environment.vars
    if "riscv_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # TODO: WARNING
        riscvInstallDir = user_vars["riscv_gcc.install_dir"]
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
    else:
        vext = params["vext"]
        pext = params["pext"]
        assert not (vext and pext)  # Combination of both extensions is currently not supported

        def _helper(url):
            fullUrlSplit = url.split("/")
            riscvUrl = "/".join(fullUrlSplit[:-1])
            riscvFileName, riscvFileExtension = fullUrlSplit[-1].split(".", 1)
            return riscvUrl, riscvFileName, riscvFileExtension

        if vext and "riscv_gcc.dl_url_vext" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars["riscv_gcc.dl_url_vext"])
        elif pext and "riscv_gcc.dl_url_pext" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars["riscv_gcc.dl_url_pext"])
        elif "riscv_gcc.dl_url" in user_vars:
            riscvUrl, riscvFileName, riscvFileExtension = _helper(user_vars["riscv_gcc.dl_url"])
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
        if (Path(riscvInstallDir) / name).is_dir():
            gccName = name
            break
    assert gccName is not None, "Toolchain name could not be dtemined automatically"
    context.cache["riscv_gcc.install_dir", flags] = riscvInstallDir
    context.cache["riscv_gcc.name", flags] = gccName


########
# llvm #
########


def _validate_llvm(context: MlonMcuContext, params=None):
    return context.environment.has_framework("tvm") or context.environment.has_target("etiss_pulpino")


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
        llvmInstallDir = user_vars["llvm.install_dir"]
    else:
        llvmVersion = user_vars.get("llvm.version", "14.0.0")
        llvmDist = user_vars.get("llvm.distribution", "x86_64-linux-gnu-ubuntu-18.04")
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
    if "dbg" in params:
        dbg = params["dbg"]
        if dbg:
            if not context.environment.has_feature("etissdbg"):
                return False
    return context.environment.has_target("etiss_pulpino")


@Tasks.provides(["etiss.src_dir"])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def clone_etiss(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the ETISS repository."""
    etissName = utils.makeDirName("etiss")
    etissSrcDir = context.environment.paths["deps"].path / "src" / etissName
    if rebuild or not utils.is_populated(etissSrcDir):
        etissRepo = context.environment.repos["etiss"]
        utils.clone(etissRepo.url, etissSrcDir, branch=etissRepo.ref)
    context.cache["etiss.src_dir"] = etissSrcDir


@Tasks.needs(["etiss.src_dir", "llvm.install_dir"])
@Tasks.provides(["etiss.build_dir", "etiss.install_dir"])
@Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etiss(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the ETISS simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.environment.paths["deps"].path / "build" / etissName
    etissInstallDir = context.environment.paths["deps"].path / "install" / etissName
    # llvmInstallDir = context.cache["llvm.install_dir"]
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
        utils.make(cwd=etissBuildDir, threads=threads, live=verbose)
    context.cache["etiss.install_dir", flags] = etissInstallDir
    context.cache["etiss.build_dir", flags] = etissBuildDir


@Tasks.needs(["etiss.build_dir"])
@Tasks.provides(["etiss.lib_dir", "etiss.install_dir", "etissvp.exe", "etissvp.script"])
@Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def install_etiss(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Install ETISS."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.cache["etiss.build_dir", flags]
    etissInstallDir = context.cache["etiss.install_dir", flags]
    etissvpExe = etissInstallDir / "bin" / "bare_etiss_processor"
    etissvpScript = etissInstallDir / "bin" / "run_helper.sh"
    etissLibDir = etissInstallDir / "lib"
    if rebuild or not utils.is_populated(etissLibDir) or not etissvpExe.is_file():
        utils.make("install", cwd=etissBuildDir, threads=threads, live=verbose)
    context.cache["etiss.lib_dir", flags] = etissLibDir
    context.cache["etiss.install_dir", flags] = etissInstallDir
    context.cache["etissvp.exe", flags] = etissvpExe
    context.cache["etissvp.script", flags] = etissvpScript


#######
# tvm #
#######


def _validate_tvm(context: MlonMcuContext, params=None):
    if "patch" in params and bool(params["patch"]):
        if not context.environment.has_feature("disable_legalize"):
            return False
    if "cmsisnn" in params and bool(params["cmsisnn"]):
        if not context.environment.has_feature("cmsisnnbyoc"):
            return False

    return context.environment.has_framework("tvm")


@Tasks.provides(["tvm.src_dir", "tvm.configs_dir"])
@Tasks.optional(["tvm_extensions.src_dir"])
@Tasks.validate(_validate_tvm)
@Tasks.param("patch", [False, True])  # This is just a temporary workaround until the patch is hopefully upstreamed
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
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
    context.cache["tvm.configs_dir", flags] = tvmSrcDir / "configs"
    context.cache["tvm.pythonpath", flags] = tvmPythonPath


@Tasks.needs(["tvm.src_dir", "llvm.install_dir"])
@Tasks.provides(["tvm.build_dir", "tvm.lib"])
@Tasks.param("dbg", False)
@Tasks.param("cmsisnn", [False, True])
@Tasks.validate(_validate_tvm)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Build the TVM framework."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"), (params["cmsisnn"], "cmsisnn"))
    dbg = bool(params["dbg"])
    cmsisnn = bool(params["cmsisnn"])
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
        cfgFileSrc = Path(tvmSrcDir) / "cmake" / "config.cmake"
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
        if cmsisnn:
            utils.exec(
                "sed",
                "-i",
                "--",
                r"s/USE_CMSISNN \(OFF\|ON\)/USE_CMSISNN ON/g",
                str(cfgFile),
            )

        utils.cmake(tvmSrcDir, cwd=tvmBuildDir, debug=dbg, use_ninja=ninja, live=verbose)
        utils.make(cwd=tvmBuildDir, threads=threads, use_ninja=ninja, live=verbose)
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
def clone_utvm_staticrt_codegen(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
def build_utvm_staticrt_codegen(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
        utils.make(cwd=utvmcgBuildDir, threads=threads, live=verbose)
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
        toolchain = params.get("toolchain", "gcc")
        if params.get("vext", False):
            if not context.environment.supports_feature("vext"):
                return False
        if params.get("pext", False):
            if toolchain == "llvm":
                # Unsupported
                return False
            if not context.environment.supports_feature("pext"):
                return False
        if params.get("vext", False) and params.get("pext", False):
            # Either pext or vext!
            return False
        # TODO: validate chosen toolchain?
    return True


@Tasks.provides(["muriscvnn.src_dir", "muriscvnn.inc_dir"])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def clone_muriscvnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
@Tasks.param("pext", [False, True])
@Tasks.param("toolchain", ["gcc"])
# @Tasks.param("target_arch", ["x86", "riscv", "arm"])  # TODO: implement
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def build_muriscvnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build muRISCV-NN."""
    if not params:
        params = {}
    flags = utils.makeFlags(
        (params["dbg"], "dbg"), (params["vext"], "vext"), (params["pext"], "pext"), (True, params["toolchain"])
    )
    flags_ = utils.makeFlags((params["vext"], "vext"), (params["pext"], "pext"))
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
        toolchain = params.get("toolchain", "gcc")
        assert gccName == "riscv32-unknown-elf" or toolchain != "llvm", "muRISCV-NN requires a non-multilib toolchain!"
        muriscvnnArgs = []
        if "riscv_gcc.install_dir" in user_vars:
            riscv_gcc = user_vars["riscv_gcc.install_dir"]
        else:
            riscv_gcc = context.cache["riscv_gcc.install_dir", flags_]
        muriscvnnArgs.append("-DRISCV_GCC_PREFIX=" + str(riscv_gcc))
        muriscvnnArgs.append("-DTOOLCHAIN=" + params["toolchain"].upper())
        vext = params.get("vext", False)
        pext = params.get("pext", False)
        muriscvnnArgs.append("-DUSE_VEXT=" + ("ON" if vext else "OFF"))
        muriscvnnArgs.append("-DUSE_PEXT=" + ("ON" if pext else "OFF"))
        muriscvnnArgs.append(f"-DRISCV_GCC_BASENAME={gccName}")
        utils.cmake(
            muriscvnnSrcDir,
            *muriscvnnArgs,
            cwd=muriscvnnBuildDir,
            debug=params["dbg"],
            live=verbose,
        )
        utils.make(cwd=muriscvnnBuildDir, threads=threads, live=verbose)
        utils.mkdirs(muriscvnnInstallDir)
        utils.move(muriscvnnBuildDir / "Source" / "libmuriscv_nn.a", muriscvnnLib)
    context.cache["muriscvnn.build_dir", flags] = muriscvnnBuildDir
    context.cache["muriscvnn.lib", flags] = muriscvnnLib


def _validate_spike(context: MlonMcuContext, params=None):
    if not context.environment.has_target("spike"):
        return False
    if params.get("vext", False):
        if params.get("pext", False):
            return False  # Can not use booth at a time
        if not context.environment.supports_feature("vext"):
            return False
    if params.get("pext", False):
        if params.get("vext", False):
            return False  # Can not use booth at a time
        if not context.environment.supports_feature("pext"):
            return False
    assert "spikepk" in context.environment.repos, "Undefined repository: 'spikepk'"
    assert "spike" in context.environment.repos, "Undefined repository: 'spike'"
    return True


@Tasks.provides(["spikepk.src_dir"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_spike_pk(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
@Tasks.param("vext", [False, True])
@Tasks.param("pext", [False, True])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def build_spike_pk(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build Spike proxy kernel."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["vext"], "vext"), (params["pext"], "pext"))
    spikepkName = utils.makeDirName("spikepk", flags=flags)
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
        vext = params.get("vext", False)
        pext = params.get("pext", False)
        assert not (pext and vext), "Currently only p or vector extension can be enabled at a time."
        if "riscv_gcc.install_dir" in user_vars:
            riscv_gcc = user_vars["riscv_gcc.install_dir"]
        else:
            riscv_gcc = context.cache["riscv_gcc.install_dir", flags]
        arch = "rv32gc"
        if pext:
            arch += "p"
        if vext:
            arch += "v"
        spikepkArgs = []
        spikepkArgs.append("--prefix=" + str(riscv_gcc))
        spikepkArgs.append("--host=" + gccName)
        spikepkArgs.append(f"--with-arch={arch}")
        spikepkArgs.append("--with-abi=ilp32d")
        env = os.environ.copy()
        env["PATH"] = str(Path(riscv_gcc) / "bin") + ":" + env["PATH"]
        utils.exec_getout(
            str(spikepkSrcDir / "configure"),
            *spikepkArgs,
            cwd=spikepkBuildDir,
            env=env,
            live=verbose,
            print_output=False,
        )
        utils.make(cwd=spikepkBuildDir, threads=threads, live=verbose, env=env)
        # utils.make(target="install", cwd=spikepkBuildDir, live=verbose, env=env)
        utils.mkdirs(spikepkInstallDir)
        utils.move(spikepkBuildDir / "pk", spikepkBin)
    context.cache["spikepk.build_dir", flags] = spikepkBuildDir
    context.cache["spike.pk", flags] = spikepkBin


@Tasks.provides(["spike.src_dir"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_spike(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
def build_spike(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
        utils.make(cwd=spikeBuildDir, threads=threads, live=verbose)
        # utils.make(target="install", cwd=spikeBuildDir, threads=threads, live=verbose)
        utils.mkdirs(spikeInstallDir)
        utils.move(spikeBuildDir / "spike", spikeExe)
    context.cache["spike.build_dir"] = spikeBuildDir
    context.cache["spike.exe"] = spikeExe


def _validate_cmsisnn(context: MlonMcuContext, params=None):
    if not (context.environment.has_feature("cmsisnn") or context.environment.has_feature("cmsisnnbyoc")):
        return False
    mvei = params.get("mvei", False)
    dsp = params.get("dsp", False)
    target_arch = params.get("target_arch", None)
    if target_arch == "arm":
        if dsp and not context.environment.has_feature("arm_dsp"):
            return False
        if mvei and not context.environment.has_feature("arm_mvei"):
            return False
    else:
        if mvei or dsp:
            return False
    return True


def _validate_cmsis(context: MlonMcuContext, params=None):
    return _validate_cmsisnn(context, params=params) or context.environment.has_target("corstone300")


@Tasks.provides(["cmsisnn.dir"])
@Tasks.validate(_validate_cmsis)
@Tasks.register(category=TaskType.MISC)
def clone_cmsis(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """CMSIS repository."""
    cmsisName = utils.makeDirName("cmsis")
    cmsisSrcDir = context.environment.paths["deps"].path / "src" / cmsisName
    # TODO: allow to skip this if cmsisnn.dir+cmsisnn.lib are provided by the user and corstone is not used
    # -> move those checks to validate?
    if rebuild or not utils.is_populated(cmsisSrcDir):
        cmsisRepo = context.environment.repos["cmsis"]
        utils.clone(cmsisRepo.url, cmsisSrcDir, branch=cmsisRepo.ref, refresh=rebuild)
    context.cache["cmsisnn.dir"] = cmsisSrcDir


@Tasks.needs(["cmsisnn.dir"])
@Tasks.optional(["riscv_gcc.install_dir", "riscv_gcc.name", "arm_gcc.install_dir"])
@Tasks.provides(["cmsisnn.lib"])
@Tasks.param("dbg", [False, True])
@Tasks.param("target_arch", ["x86", "riscv", "arm"])
@Tasks.param("mvei", [False, True])
@Tasks.param("dsp", [False, True])
@Tasks.validate(_validate_cmsisnn)
@Tasks.register(category=TaskType.OPT)  # TODO: rename to TaskType.FEATURE?
def build_cmsisnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    target_arch = params["target_arch"]
    mvei = params["mvei"]
    dsp = params["dsp"]
    dbg = params["dbg"]
    flags = utils.makeFlags(
        (True, target_arch),
        (mvei, "mvei"),
        (dsp, "dsp"),
        (dbg, "dbg"),
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
            # Warning: this does not work!
            if dsp:
                cmakeArgs.append("-DARM_MATH_DSP=ON")
            if mvei:
                cmakeArgs.append("-DARM_MATH_MVEI=ON")
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
            raise ValueError(f"Target architecture '{target_arch}' is not supported")

        utils.cmake(
            *cmakeArgs,
            str(cmsisnnSrcDir),
            debug=dbg,
            cwd=cmsisnnBuildDir,
            live=verbose,
            env=env,
        )
        utils.make(cwd=cmsisnnBuildDir, threads=threads, live=verbose)
        utils.mkdirs(cmsisnnInstallDir)
        utils.move(cmsisnnBuildDir / "Source" / "libcmsis-nn.a", cmsisnnLib)
    context.cache["cmsisnn.lib", flags] = cmsisnnLib


def _validate_corstone300(context: MlonMcuContext, params=None):
    return context.environment.has_target("corstone300")


@Tasks.provides(["arm_gcc.install_dir"])
@Tasks.validate(_validate_corstone300)
@Tasks.register(category=TaskType.TARGET)
def install_arm_gcc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install GNU compiler toolchain from ARM."""
    armName = utils.makeDirName("arm_gcc")
    armInstallDir = context.environment.paths["deps"].path / "install" / armName
    user_vars = context.environment.vars
    if "arm_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # armInstallDir = user_vars["riscv_gcc.install_dir"]
        return False
    else:
        if not utils.is_populated(armInstallDir):
            armUrl = "https://developer.arm.com/-/media/Files/downloads/gnu/11.2-2022.02/binrel/"
            armFileName = "gcc-arm-11.2-2022.02-x86_64-arm-none-eabi"
            armArchive = armFileName + ".tar.xz"
            utils.download_and_extract(armUrl, armArchive, armInstallDir)
    context.cache["arm_gcc.install_dir"] = armInstallDir


@Tasks.provides(["corstone300.exe"])
@Tasks.validate(_validate_corstone300)
@Tasks.register(category=TaskType.TARGET)
def install_corstone300(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
def clone_tvm_extensions(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
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
def clone_mlif(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the MLonMCU SW repository."""
    mlifName = utils.makeDirName("mlif")
    mlifSrcDir = context.environment.paths["deps"].path / "src" / mlifName
    if rebuild or not utils.is_populated(mlifSrcDir):
        mlifRepo = context.environment.repos["mlif"]
        utils.clone(mlifRepo.url, mlifSrcDir, branch=mlifRepo.ref, refresh=rebuild)
    context.cache["mlif.src_dir"] = mlifSrcDir


def _validate_espidf(context: MlonMcuContext, params=None):
    return context.environment.has_platform("espidf")


@Tasks.provides(["espidf.src_dir"])
@Tasks.validate(_validate_espidf)
@Tasks.register(category=TaskType.PLATFORM)
def clone_espidf(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the ESP-IDF repository."""
    espidfName = utils.makeDirName("espidf")
    espidfSrcDir = context.environment.paths["deps"].path / "src" / espidfName
    user_vars = context.environment.vars
    if "espidf.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(espidfSrcDir):
        espidfRepo = context.environment.repos["espidf"]
        utils.clone(espidfRepo.url, espidfSrcDir, branch=espidfRepo.ref, refresh=rebuild, recursive=True)
    context.cache["espidf.src_dir"] = espidfSrcDir


@Tasks.needs(["espidf.src_dir"])
@Tasks.provides(["espidf.install_dir"])
@Tasks.validate(_validate_espidf)
@Tasks.register(category=TaskType.PLATFORM)
def install_espidf(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install target support for ESP-IDF toolchain."""
    espidfName = utils.makeDirName("espidf")
    espidfInstallDir = context.environment.paths["deps"].path / "install" / espidfName
    espidfSrcDir = context.cache["espidf.src_dir"]  # TODO: This will fail if the espidf.src_dir is user-supplied
    user_vars = context.environment.vars
    if "espidf.install_dir" in user_vars:  # TODO: also check command line flags?
        return False
    boards = ["all"]
    if "espidf.boards" in user_vars:
        boards = user_vars["espidf.boards"]
    if not isinstance(boards, str):
        assert isinstance(boards, list)
        boards = ",".join(boards)
    if not utils.is_populated(espidfInstallDir) or rebuild:
        # Using idf_tools.py directory instead of ./install.sh because we
        # don't want to use espe-idfs python environment
        espidfInstallScript = Path(espidfSrcDir) / "tools" / "idf_tools.py"
        espidfInstallArgs = ["install", f"--targets={boards}"]
        env = os.environ.copy()
        env["IDF_TOOLS_PATH"] = str(espidfInstallDir)
        utils.python(espidfInstallScript, *espidfInstallArgs, print_output=False, live=verbose, env=env)
    context.cache["espidf.install_dir"] = espidfInstallDir


def _validate_tflite_visualize(context: MlonMcuContext, params=None):
    return context.environment.has_frontend("tflite") and context.environment.has_feature("visualize")


@Tasks.provides(["tflite_visualize.exe"])
@Tasks.validate(_validate_tflite_visualize)
@Tasks.register(category=TaskType.FEATURE)
def download_tflite_vizualize(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download the visualize.py script for TFLite."""
    # This script is content of the tensorflow repo (not tflite-micro) and unfortunately not bundled
    # into the tensorflow python package. Therefore just download this single file form GitHub

    tfLiteVizualizeName = utils.makeDirName("tflite_visualize")
    tfLiteVizualizeInstallDir = context.environment.paths["deps"].path / "install" / tfLiteVizualizeName
    tfLiteVizualizeExe = tfLiteVizualizeInstallDir / "visualize.py"
    user_vars = context.environment.vars
    if "tflite_visualize.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(tfLiteVizualizeInstallDir):
        tfLiteVizualizeInstallDir.mkdir()
        url = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/visualize.py"
        utils.download(url, tfLiteVizualizeExe)
    context.cache["tflite_visualize.exe"] = tfLiteVizualizeExe


def _validate_microtvm_etissvp(context: MlonMcuContext, params=None):
    return context.environment.has_feature("microtvm_etissvp")


@Tasks.provides(["microtvm_etissvp.src_dir", "microtvm_etissvp.template"])
@Tasks.validate(_validate_microtvm_etissvp)
@Tasks.register(category=TaskType.FEATURE)
def clone_microtvm_etissvp(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the microtvm-etissvp-template repository."""
    name = utils.makeDirName("microtvm_etissvp")
    srcDir = context.environment.paths["deps"].path / "src" / name
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["microtvm_etissvp"]
        utils.clone(repo.url, srcDir, branch=repo.ref, refresh=rebuild)
    context.cache["microtvm_etissvp.src_dir"] = srcDir
    context.cache["microtvm_etissvp.template"] = srcDir / "template_project"
