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
import tempfile
import tarfile
from pathlib import Path
from typing import Tuple
import multiprocessing

from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from mlonmcu.timeout import exec_timeout
from mlonmcu.config import str2bool, str2list, str2dict
from mlonmcu.logging import get_logger
from .model_info import get_model_info, get_fallback_model_info, get_supported_formats, get_model_format
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat
from .python_utils import prepare_python_environment
from .tvmc_utils import (
    get_target_tvmc_args,
    get_pass_config_tvmc_args,
    get_disabled_pass_tvmc_args,
    get_runtime_executor_tvmc_args,
    get_input_shapes_tvmc_args,
    get_tuning_records_tvmc_args,
    get_desired_layout_args,
)

logger = get_logger()


class TVMBackend(Backend):
    registry = {}

    name = None

    FEATURES = {"autotuned", "cmsisnnbyoc", "muriscvnnbyoc", "disable_legalize", "moiopt", "uma_backends", "fuse_ops"}

    DEFAULTS = {
        "print_outputs": False,
        "opt_level": 3,
        "target_device": None,
        "target_mcpu": None,
        "target_march": None,
        "target_model": None,
        "target_mtriple": None,
        "target_mabi": None,
        "target_mattr": None,
        "target_keys": None,
        "target_num_cores": None,
        "extra_targets": None,  # list
        "extra_target_details": None,  # dict
        "desired_layout": None,  # optional: NCHW, NHWC, NHWC:HWOI, ...
        "desired_layout_ops": None,  # optional: conv2d, max_pool2d,...
        "desired_layout_map": None,  # optional, conv2d=NCHW, ...
        "disabled_passes": [],  # i.e. AlterOpLayout
        "extra_pass_config": {},  # TODO: some example (fuse_max_depth etc.)
        "use_tuning_results": False,
        "tvmc_extra_args": [],  # Currently compile subcommand only!
        "tvmc_custom_script": None,
        # See https://github.com/apache/tvm/blob/1115fd9bc261619ffa0539746ae0aebc46232dc6/python/tvm/autotvm/tophub.py
        "tophub_url": None,
        "num_threads": multiprocessing.cpu_count(),
        "dump": [],  # Supports: c, relay, tir, ll
        "disable_vectorize": "auto",
        "custom_unroll": False,  # Experimental, RISC-V only
        "autotuned_mode": None,
        "autotuned_results_file": None,
        "relay_debug": None,  # Use "DEFAULT=2" to have most verbosity. Needs USE_RELAY_DEBUG during setup.
        "refresh_model_info": False,
        "generate_wrapper": "auto",
    }

    REQUIRED = set()

    OPTIONAL = {"tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir", "tvm.use_tlcpack"}

    def __init__(
        self, target="c", executor=None, runtime="crt", fmt="mlf", system_lib=False, features=None, config=None
    ):
        super().__init__(framework="tvm", features=features, config=config)

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None
        self.model_format = None
        self.supported_formats = get_supported_formats()
        self.target = target
        self.runtime = runtime
        self.executor = executor
        self.fmt = fmt
        self.system_lib = system_lib

        self.prefix = "default"
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        self._tuning_records = {}
        results_file = self.config.get("autotuned_results_file", None)
        tuner_name = self.config.get("autotuned_mode", None)
        if results_file is not None:
            assert tuner_name is not None
            self._tuning_records[tuner_name] = results_file

    # On the long term, we might support multiple TUNE stages in a single run
    # (i.e. to allow autotvm+graphtuner to be separated)
    # Hence
    # @property
    # def tuning_records(self):
    #     if self._tuning_records:
    #         return self._tuning_records
    #     return self.config.get("autotuning_results_file", None):

    # @tuning_records.setter
    # def tuning_records(self, filepath):
    #     self._tuning_records = filepath

    def set_tuning_records(self, records, tuner_name=None):
        if tuner_name is None:
            tuner_name = self.config["autotuned_mode"]
            # tuner_name = "autotvm"
            assert tuner_name is not None
        self._tuning_records[tuner_name] = records

    def get_tuning_records(self, tuner_name=None):
        if tuner_name is None:
            tuner_name = self.config["autotuned_mode"]
            # tuner_name = "autotvm"
            if tuner_name is None:
                if len(self._tuning_records) > 0:
                    tuner_name = list(self._tuning_records.keys())[0]
        return self._tuning_records.get(tuner_name, None)

    @property
    def disable_vectorize(self):
        temp = self.config["disable_vectorize"]
        if temp is None or (isinstance(temp, str) and temp in ["auto", "AUTO"]):
            return self.target == "c"
        return str2bool(temp)

    @property
    def extra_pass_config(self):
        extra = self.config["extra_pass_config"]
        if extra is None:
            extra = {}
        if isinstance(extra, str):
            import ast

            extra = ast.literal_eval(extra)
        assert isinstance(extra, dict)
        return extra

    @property
    def pass_config(self):
        base = {"tir.disable_vectorize": self.disable_vectorize}
        extra = self.extra_pass_config
        base.update(extra)
        return base

    @property
    def target_device(self):
        return self.config["target_device"]

    @property
    def target_mcpu(self):
        return self.config["target_mcpu"]

    @property
    def target_march(self):
        return self.config["target_march"]

    @property
    def target_mtriple(self):
        return self.config["target_mtriple"]

    @property
    def target_mabi(self):
        return self.config["target_mabi"]

    @property
    def target_mattr(self):
        return self.config["target_mattr"]

    @property
    def target_keys(self):
        return self.config["target_keys"]

    @property
    def target_model(self):
        return self.config["target_model"]

    @property
    def target_num_cores(self):
        return self.config["target_num_cores"]

    # TODO:
    # "target_device": ?,
    # "target_libs": ?,
    # "target_tag": ?,
    # "target_march": ?,
    # "target_keys": ?,
    # "target_opt_level": ?,
    # "target_cl_opt": ?,
    # "target_mfloat_abi": ?,
    # "target_fast_math_ninf": ?,
    # "target_fast_math_contract": ?,
    # "target_fast_math_nnan": ?,
    # "target_fast_math": ?,
    # "target_fast_math_nsz": ?,
    # "target_fast_math_reassoc": ?,
    # "target_fast_math_arcp": ?,

    @property
    def extra_targets(self):
        return str2list(self.config["extra_targets"], allow_none=True)

    @property
    def extra_target_details(self):
        return str2dict(self.config["extra_target_details"], allow_none=True)

    @property
    def desired_layout(self):
        return str2list(self.config["desired_layout"], allow_none=True)

    @property
    def desired_layout_ops(self):
        return str2list(self.config["desired_layout_ops"], allow_none=True)

    @property
    def desired_layout_map(self):
        return str2dict(self.config["desired_layout_map"], allow_none=True)

    @property
    def opt_level(self):
        return self.config["opt_level"]

    @property
    def use_tuning_results(self):
        value = self.config["use_tuning_results"]
        return str2bool(value)

    @property
    def tvmc_extra_args(self):
        return self.config["tvmc_extra_args"]

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

    @property
    def disabled_passes(self):
        value = self.config["disabled_passes"]
        return str2list(value)

    @property
    def tvm_pythonpath(self):
        return self.config["tvm.pythonpath"]

    @property
    def tvm_build_dir(self):
        return self.config["tvm.build_dir"]

    @property
    def tvm_configs_dir(self):
        return self.config["tvm.configs_dir"]

    @property
    def tophub_url(self):
        return self.config["tophub_url"]

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    @property
    def use_tlcpack(self):
        value = self.config["tvm.use_tlcpack"]
        return str2bool(value, allow_none=True)

    @property
    def custom_unroll(self):
        value = self.config["custom_unroll"]
        return str2bool(value, allow_none=True)

    @property
    def dump(self):
        value = self.config["dump"]
        if isinstance(value, str):
            if "," in value:
                value = value.split(",")
            else:
                value = [value]
        for v in value:
            assert v in ["relay", "c", "ll", "tir"]
        assert isinstance(value, list)
        return value

    @property
    def needs_target(self):
        return self.target == "llvm"  # not c

    @property
    def refresh_model_info(self):
        value = self.config["refresh_model_info"]
        return str2bool(value, allow_none=True)

    @property
    def generate_wrapper(self):
        value = self.config["generate_wrapper"]
        if isinstance(value, str):
            if value.lower() == "auto":
                value = self.fmt == "mlf"
            else:
                value = str2bool(value)
        assert isinstance(value, bool)
        return value

    @property
    def num_threads(self):
        return self.config["num_threads"]

    @property
    def relay_debug(self):
        return self.config["relay_debug"]

    def get_target_details(self):
        ret = {}
        if self.target_device:
            ret["device"] = self.target_device
        if self.target_mcpu:
            ret["mcpu"] = self.target_mcpu
        if self.target_march:
            ret["march"] = self.target_march
        if self.target_mtriple:
            ret["mtriple"] = self.target_mtriple
        if self.target_mabi:
            ret["mabi"] = self.target_mabi
        if self.target_mattr:
            temp = self.target_mattr
            if self.custom_unroll:
                temp += ",+no-default-unroll"
            ret["mattr"] = temp
        if self.target_keys:
            ret["keys"] = self.target_keys
        if self.target_model:
            ret["model"] = self.target_model
        if self.target_num_cores:
            ret["num-cores"] = self.target_num_cores
        return ret

    def get_tvmc_compile_args(self, out, dump=None):
        assert self.executor is not None
        assert self.executor in ["aot", "graph"], "Unsupported TVM executor"
        args = [
            self.model,
            *get_target_tvmc_args(
                self.target,
                extra_targets=self.extra_targets,
                target_details=self.get_target_details(),
                extra_target_details=self.extra_target_details,
            ),
            *get_runtime_executor_tvmc_args(self.runtime, self.executor),
            *get_pass_config_tvmc_args(self.pass_config),
            *get_disabled_pass_tvmc_args(self.disabled_passes),
            *get_input_shapes_tvmc_args(self.input_shapes),
            *get_tuning_records_tvmc_args(self.use_tuning_results, self.get_tuning_records()),
            *get_desired_layout_args(self.desired_layout, self.desired_layout_ops, self.desired_layout_map),
            *(["--dump-code", ",".join(dump)] if dump is not None and len(dump) > 0 else []),
            *self.tvmc_extra_args,
            *["--opt-level", str(self.opt_level)],
            *["--output", str(out)],
            *["-f", self.fmt],
            *["--model-format", self.model_format],
        ]
        return args

    def invoke_tvmc(self, command, *args, cwd=None):
        env = prepare_python_environment(
            None if self.use_tlcpack else self.tvm_pythonpath,
            None if self.use_tlcpack else self.tvm_build_dir,
            None if self.use_tlcpack else self.tvm_configs_dir,
            tophub_url=self.tophub_url,
            num_threads=self.num_threads,
            debug_cfg=self.relay_debug,
        )
        if self.use_tlcpack:
            pre = ["tvmc"]
            return utils.execute(*pre, command, *args, live=self.print_outputs, env=env, cwd=cwd)
        else:
            if self.tvmc_custom_script is None:
                pre = ["-m", "tvm.driver.tvmc"]
            else:
                pre = [self.tvmc_custom_script]
            return utils.python(*pre, command, *args, live=self.print_outputs, env=env, cwd=cwd)

    def invoke_tvmc_compile(self, out, dump=None, cwd=None):
        args = self.get_tvmc_compile_args(out, dump=dump)
        # self.timeout_sec = 90
        self.timeout_sec = 0
        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                self.invoke_tvmc,
                "compile",
                *args,
                cwd=cwd,
            )
        else:
            ret = self.invoke_tvmc("compile", *args, cwd=cwd)
        return ret

    def load_model(self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None):
        self.model = model
        # TODO: path model class instead of path!
        # fmt = self.model.formats[0]
        need_model_info = True
        if input_shapes:
            self.input_shapes = input_shapes
            if output_shapes and input_types and output_types:
                need_model_info = False
                self.model_format, self.model_info = get_fallback_model_info(
                    model, input_shapes, output_shapes, input_types, output_types, backend_name=self.name
                )
        else:
            self.input_shapes = None  # Relevant for multiple subs using the same backend
        if need_model_info:
            try:
                self.model_format, self.model_info = get_model_info(model, backend_name=self.name)
            except Exception as e:
                self.model_format = get_model_format(model)
                if self.model_format != "relay":
                    logger.warning(
                        "Fetching of Model Info failed (%s). Falling back to Relay-based info.", type(e).__name__
                    )
                    self.model_info = None
                else:
                    raise e

            if self.model_info:
                # TODO: also handle output_shapes
                # TODO: take care of refresh_model_info
                if self.input_shapes:
                    self.model_info.in_tensors = [t for t in self.model_info.in_tensors if t.name in self.input_shapes]
                    assert (
                        len(self.model_info.in_tensors) > 0
                    ), "Missmatch between provided input names and detected ones"
                else:
                    self.input_shapes = {tensor.name: tensor.shape for tensor in self.model_info.in_tensors}
        if self.model_info:
            self.model_info.validate()

    def get_graph_and_params_from_mlf(self, path):
        graph = None
        with open(Path(path) / "executor-config" / "graph" / "default.graph", "r") as handle:
            graph = handle.read()
        params = None
        with open(Path(path) / "parameters" / "default.params", "rb") as handle:
            params = handle.read()

        return graph, params

    def generate(self) -> Tuple[dict, dict]:
        artifacts = []
        assert self.model is not None
        dump = self.dump
        if self.refresh_model_info or (self.generate_wrapper and not self.model_info) and "relay" not in dump:
            dump.append("relay")
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / f"{self.prefix}.tar"
            out = self.invoke_tvmc_compile(out_path, dump=dump, cwd=temp_dir)
            if self.fmt == "mlf":
                mlf_path = Path(temp_dir) / "mlf"
                tarfile.open(out_path).extractall(mlf_path)
                with open(mlf_path / "metadata.json", "r") as handle:
                    metadata_txt = handle.read()
                artifacts.append(
                    Artifact(
                        f"{self.prefix}.json",
                        content=metadata_txt,
                        fmt=ArtifactFormat.TEXT,
                    )
                )
            with open(out_path, "rb") as handle:
                data = handle.read()
                artifacts.append(
                    Artifact(
                        f"{self.prefix}.tar",
                        raw=data,
                        fmt=ArtifactFormat.SHARED_OBJECT if self.fmt == "so" else ArtifactFormat.MLF,
                        archive=True,
                    )
                )
            if "c" in dump:
                with open(str(out_path) + ".c", "r") as handle:
                    mod_src = handle.read()
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.c",
                            content=mod_src,
                            fmt=ArtifactFormat.SOURCE,
                            optional=True,
                        )
                    )
            if "relay" in dump:
                with open(str(out_path) + ".relay", "r") as handle:
                    mod_txt = handle.read()
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.relay",
                            content=mod_txt,
                            fmt=ArtifactFormat.TEXT,
                            optional=True,
                        )
                    )
            if "ll" in dump:
                with open(str(out_path) + ".ll", "r") as handle:
                    mod_txt = handle.read()
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.ll",
                            content=mod_txt,
                            fmt=ArtifactFormat.SOURCE,
                            optional=True,
                        )
                    )
            if self.executor == "graph":
                if self.fmt == "so":
                    pass
                    # raise NotImplementedError
                elif self.fmt == "mlf":
                    graph, params = self.get_graph_and_params_from_mlf(mlf_path)
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.graph",
                            content=graph,
                            fmt=ArtifactFormat.SOURCE,
                        )
                    )
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.params",
                            raw=params,
                            fmt=ArtifactFormat.RAW,
                        )
                    )
                else:
                    raise RuntimeError("Unsupported fmt")
        stdout_artifact = Artifact(
            "tvmc_compile_out.log", content=out, fmt=ArtifactFormat.TEXT
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": Metrics()}
