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
import os
import sys
import logging
import shutil
import argparse
import tarfile
import json
import re
from typing import Optional
import yaml
import subprocess
import pathlib

import numpy as np

# from matplotlib import pyplot as plt

# import tvm
# import tvm.micro
# from tvm import te
# from tvm import relay
# from tvm import ir
# from tvm import autotvm
# from tvm.contrib import graph_runtime
# from tvm.contrib import utils as tvm_utils
# from tvm.micro import export_model_library_format
#
## import compiler_riscv
# import codegen
# from load_tflite_model import load_tflite_model
# from ftp import FTPass
# from plan_memory import plan_memory
#
# from tvm.ir import _ffi_api as ir_ffi

import mlonmcu.cli.helper.filter as cli_filter

# import mlonmcu.flow.tvm.wrapper as tvm_wrapper

# from mlonmcu.flow.tvm.transform.reshape import ReshapeInfo, RemoveReshapeOnlyPass, FixReshapesPass
# from mlonmcu.flow.tvm.transform.attrs import CheckAttrs
# from mlonmcu.flow.tvm.transform.legalize import OptionallyDisableLegalize

# Setup logging
from mlonmcu.logging import get_logger

logger = get_logger()
logging.getLogger("compile_engine").setLevel(logging.WARNING)
logging.getLogger("autotvm").setLevel(logging.WARNING)


# class TVMFlow:
#    def __init__(
#        self,
#        local=False,
#        transformations=[],
#        aot=False,
#        unpacked=False,
#        outDir=os.path.join(os.getcwd(), "out"),
#        verbose=False,
#        fuseMaxDepth=-1,
#        arenaBytes=-1,
#    ):
#        self.opt_level = 3
#        self.local = local
#        self.transformations = transformations
#        self.useAOT = aot
#        # Unpacked API is incompatible with most BYOC kernels and not supported by graph runtime!
#        self.unpacked = "unpacked_api" in self.transformations
#        targetStr = "llvm" if self.local else "c"
#        targetStr += " --runtime=c"
#        if "arm_schedules" in self.transformations:
#            targetStr += " -device=arm_cpu"
#        if self.useAOT:
#            targetStr += " --link-params"
#            targetStr += " --executor=aot"
#            targetStr += " --workspace-byte-alignment={}".format(4)
#            targetStr += " --unpacked-api={}".format(int(self.unpacked))
#            targetStr += " --interface-api={}".format("c" if self.unpacked else "packed")
#        else:
#            # TODO -link-params: adds static parameters into generated code
#            targetStr += " -model=unknown --system-lib"
#        self.target = tvm.target.Target(targetStr)
#        self.outDir = outDir
#        self.verbose = verbose
#        self.fuseMaxDepth = fuseMaxDepth
#        self.arenaBytes = arenaBytes
#        self.modName = "model"
#
#    def loadModel(self, path):
#        logger.info("### TVMFlow.loadModel")
#
#        modelBuf = open(path, "rb").read()
#        self.mod, self.params, self.modelInfo = load_tflite_model(modelBuf)
#
#        if "fusetile" in self.transformations or "memplan" in self.transformations:
#            modelName = re.search(r"\/(\w+)\.tflite", path).group(1)
#
#            @tvm.register_func("relay.backend.PostPass")
#            def _post_pass(mod, params):
#                # CheckAttrs().visit(mod["main"])
#                # if "memplan" in self.transformations:
#                #    mod = RemoveReshapeOnlyPass()(mod)
#                mod = FixReshapesPass()(mod)
#                if "fusetile" in self.transformations:
#                    hintAxis = None
#                    if modelName == "vww":
#                        hintAxis = 3
#                    mod = FTPass(params, 2, hintAxis)(mod)
#                    # TODO: automate
#                    """
#                    if modelName == "dummy_add":
#                        mod = FTPass(params, 2, [3, 3, 3, 1, 3])(mod)
#                    elif modelName == "bigsine_quant":
#                        #mod = FTPass(params, 2, [2, 1, 1])(mod)
#                        mod = FTPass(params, 2, [0, 1, 1])(mod)
#                    """
#                mod = FixReshapesPass()(mod)
#                return mod
#
#        if "memplan" in self.transformations:
#
#            @tvm.register_func("tvm.relay.plan_memory")
#            def _plan_memory(func):
#                return plan_memory(func)
#
#        cfg = {}
#        transformations = []
#        transformations.append(relay.transform.InferType())
#        # Will be helpful when further (optional) transformations are added -> BYOC
#
#        if self.verbose:
#            logger.info("Relay Model before transformations: \n")
#            logger.info(self.mod)
#        seq = tvm.transform.Sequential(transformations)
#        with tvm.transform.PassContext(opt_level=3, config=cfg):
#            self.mod = seq(self.mod)
#
#        if self.verbose:
#            logger.info("Relay Model after transformations: \n")
#            logger.info(self.mod)
#
#    def build(self, export_only=False):
#        logger.info("### TVMFlow.build")
#
#        if self.local:
#            cfg = {}
#        else:
#            cfg = {"tir.disable_vectorize": True}
#            if self.fuseMaxDepth >= 0:
#                cfg["relay.FuseOps.max_depth"] = self.fuseMaxDepth
#
#        with tvm.transform.PassContext(opt_level=self.opt_level, config=cfg):
#            with OptionallyDisableLegalize("disable_legalize" in self.transformations):
#                c_mod = relay.build(self.mod, self.target, params=self.params, mod_name=self.modName)
#            if not self.useAOT:
#                self.graph = c_mod.get_graph_json()
#            else:
#                self.graph = None
#            self.c_params = c_mod.get_params()
#            # Individual steps:
#            # opt_mod, opt_params = relay.optimize(self.mod, self.target, self.params)
#            # grc = graph_executor_codegen.GraphExecutorCodegen(None, self.target)
#            # self.graph, lowered_mod, lowered_params = grc.codegen(opt_mod["main"])
#            # TODO: call into backend, see target.build.c / driver_api.cc
#            # print(lowered_mod)
#
#        if self.verbose:
#            # print(c_mod.get_source())
#            for k, v in self.params.items():
#                logger.debug(k, v.shape)
#            for k, v in self.c_params.items():
#                logger.debug(k, v.shape)
#
#        if not self.local:
#            # Extract metadata.
#            mlfDir = tvm_utils.tempdir().temp_dir
#            os.makedirs(mlfDir, exist_ok=True)
#            tarFile = os.path.join(mlfDir, "archive.tar")
#            export_model_library_format(c_mod, tarFile)
#            tarfile.open(tarFile).extractall(mlfDir)
#            with open(os.path.join(mlfDir, "metadata.json")) as f:
#                metadata = json.load(f)
#
#            if not export_only:
#                # TODO: fix this?
#                repo_root = pathlib.Path(
#                    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding="utf-8").strip()
#                )
#                template_project_path = repo_root / "src" / "runtime" / "crt" / "host"
#                project_options = {}
#                temp_dir = tvm_utils.tempdir()
#                generated_project_dir = temp_dir / "generated-project"
#                generated_project = tvm.micro.generate_project(
#                    template_project_path, c_mod, generated_project_dir, project_options
#                )
#                generated_project.build()
#
#            if os.path.exists(os.path.join(self.outDir, "params.bin")):
#                shutil.rmtree(self.outDir)
#
#            shutil.copytree(os.path.join(mlfDir, "codegen", "host", "src"), self.outDir)
#            shutil.copy2(os.path.join(mlfDir, "src", "relay.txt"), os.path.join(self.outDir, "relay.txt"))
#            shutil.copy2(os.path.join(mlfDir, "metadata.json"), os.path.join(self.outDir, "metadata.json"))
#
#            if self.graph:
#                with open(os.path.join(self.outDir, "graph.json"), "w") as f:
#                    f.write(self.graph)
#
#            with open(os.path.join(self.outDir, "metadata.json")) as json_f:
#                metadata = json.load(json_f)
#
#            with open(os.path.join(self.outDir, "params.bin"), "wb") as f:
#                f.write(relay.save_param_dict(self.c_params))
#
#            if self.useAOT:
#                if self.unpacked:
#                    shutil.copy2(
#                        os.path.join(mlfDir, "codegen", "host", "include", f"tvmgen_{self.modName}.h"),
#                        os.path.join(self.outDir, f"tvmgen_{self.modName}.h"),
#                    )
#                workspaceBytes = metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]
#                wrapper_codegen = tvm_wrapper.AOTWrapper(
#                    self.modelInfo,
#                    workspaceBytes,
#                    self.modName,
#                    modApi="c" if self.unpacked else "packed",
#                )
#            else:
#                # THe following does NOT work for the graph runtime because the memory requirements for JSON parsing are
#                # not considered here.
#                # workspaceBytes = metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]
#                # Instead let the user pass a workspace size and fall back to default (16 MB) if not specified
#                if self.arenaBytes >= 0:
#                    workspaceBytes = self.arenaBytes
#                else:
#                    workspaceBytes = 2 ** (24)
#                    # Get size for code generator: Max of op functions usage.
#                maxOpWorkspaceBytes = 0
#                for op in metadata["memory"]["functions"]["operator_functions"]:
#                    maxOpWorkspaceBytes = max(
#                        maxOpWorkspaceBytes,
#                        op["workspace"][0]["workspace_size_bytes"] if len(op["workspace"]) > 0 else 0,
#                    )
#                with open(os.path.join(self.outDir, "max_op_workspace_size.txt"), "w") as f:
#                    f.write(str(maxOpWorkspaceBytes))
#
#                codegen = tvm_wrapper.RTWrapper(
#                    os.path.join(self.outDir, "runtime_wrapper.c"),
#                    self.graph,
#                    relay.save_param_dict(self.c_params),
#                    self.modelInfo,
#                    workspaceBytes,
#                )
#            codegen.generateTargetCode(self.outDir, wrapper=f"{self.codegen}_wrapper.c")
#
#    def run(self, inData=None):
#        logger.info("### TVMFlow.run")
#
#        raise NotImplementedError


def get_parser():
    # TODO: add help strings should start with a lower case letter
    parser = argparse.ArgumentParser(
        description="Run TVM Flow",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Use environment variables to overwrite default paths:
      - RISCV_DIR (default: /usr/local/research/projects/SystemDesign/tools/riscv/current, read only)
""",
    )
    parser.add_argument("model", metavar="MODEL", type=str, nargs=1, help="Model to process")
    # TODO: support non-.tflite models!
    parser.add_argument(
        "--codegen",
        "-c",
        metavar="CODEGEN",
        type=str,
        nargs=1,
        choices=["aot", "graph"],
        default="aot",
        help="Choose a code generator or executor/runtime (default: %(default)s, choices: %(choices)s)",
    )
    parser.add_argument(
        "--target",
        "-t",
        metavar="TARGET",
        type=str,
        nargs=1,
        choices=["host", "arm_cpu"],
        default="host",
        help="Choose a target (default: %(default)s, choices: %(choices)s)",
    )
    # TODO: allow non-micro targets?
    parser.add_argument(
        "--output",
        "-o",
        dest="outdir",
        metavar="DIR",
        type=str,
        default=os.path.join(os.getcwd(), "out"),
        help="""Output directory (default: %(default)s)""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )
    # TODO: allow multiple -v's?

    tvm_group = parser.add_argument_group("TVM General")
    tvm_group.add_argument(
        "--wrapper",
        action="store_true",
        help="Generate a wrapper entry point (default: %(default)s)",
    )
    tvm_group.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with TVM (default: %(default)s)",
    )
    tvm_group.add_argument(
        "--run",
        action="store_true",
        help="Run model after building (default: %(default)s)",
    )
    tvm_group.add_argument(
        "--transformations",
        metavar="T",
        type=str,
        default="",
        help="""Comma-separated list of transformations to apply (default: %(default)s)
Available transformations:
  - fusetile
  - memplan
  - arm_schedules
  - disable_legalize
  - unpacked_api""",
    )
    tvm_group.add_argument(
        "--pass-config",
        action="append",
        metavar=("name=value"),
        help="configurations to be used at compile time. This option can be provided multiple "
        "times, each one to set one configuration value, "
        "e.g. '--pass-config relay.FuseOps.max_depth=1'.",
    )
    tvm_group.add_argument(
        "--tune",
        action="store_const",
        const=True,
        default=False,
        help="Use TVM Autotuner, optionally pass path to existing autotuning logs (default: %(default)s)",
    )

    graph_group = parser.add_argument_group("TVM Graph Runtime")
    graph_group.add_argument(
        "--arena-bytes",
        dest="arena_bytes",
        metavar="BYTES",
        type=int,
        default=-1,
        help="Overwrite default/determined arena size (default: %(default)s))",
    )

    aot_group = parser.add_argument_group("TVM AoT Runtime")

    return parser


def main():

    args = get_parser().parse_args()

    # TODO: setup verbosity
    # logging.basicConfig(level=logging.INFO)

    # TODO: make helper function
    # filter removes elements that evaluate to False.
    # transformations = list(filter(None, args.transformations.split(",")))
    transformations = cli_filter.filter_arg(args.transformations)

    flow = TVMFlow(
        local=args.local,
        transformations=transformations,
        aot=args.aot,
        outDir=args.outdir,
        verbose=args.verbose,
        fuseMaxDepth=args.max_depth,
        arenaBytes=args.arena_bytes,
    )

    flow.loadModel(args.model)
    if args.transform:
        flow.transform(args.transform)
    flow.build()
    if args.compile:
        flow.compile()
    if args.out:
        flow.export(wrapper=args.wrapper)
    if args.run:
        flow.run()
    if args.tune:
        flow.tune()


if __name__ == "__main__":
    main()
