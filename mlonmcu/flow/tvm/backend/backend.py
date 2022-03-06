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

# from ..support.load_tflite_model import load_tflite_model
from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from .tflite_model_info import get_tflite_model_info
from .tuner import TVMTuner


class TVMBackend(Backend):

    registry = {}

    name = None

    FEATURES = ["autotune", "autotuned", "cmsisnnbyoc", "disable_legalize"]

    DEFAULTS = {
        "print_outputs": False,
        "opt_level": 3,
        "target_device": None,
        "extra_target": None,
        "disabled_passes": [],  # i.e. AlterOpLayout
        "extra_pass_config": {},  # TODO: some example (fuse_max_depth etc.)
        "use_tuning_results": False,
        "tvmc_extra_args": [],  # Currently compile subcommand only!
        "tvmc_custom_script": None,
        **{("autotuning_" + key): value for key, value in TVMTuner.DEFAULTS.items()},
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(framework="tvm", features=features, config=config, context=context)

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None

        self.prefix = "default"
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data
        self.verbose = bool(self.config["print_outputs"])
        tuner_config = {  # This would be more compact with a helper function but for now its fine...
            "enable": self.config["autotuning_enable"],
            "results_file": self.config["autotuning_results_file"],
            "append": self.config["autotuning_append"],
            "tuner": self.config["autotuning_tuner"],
            "trials": self.config["autotuning_trials"],
            "early_stopping": self.config["autotuning_early_stopping"],
            "num_workers": self.config["autotuning_num_workers"],
            "max_parallel": self.config["autotuning_max_parallel"],
            "use_rpc": self.config["autotuning_use_rpc"],
            "timeout": self.config["autotuning_timeout"],
            "print_outputs": self.config["print_outputs"],
        }
        self.tuner = TVMTuner(self, config=tuner_config)
        self.tuning_records_file = None

    def set_tuning_records(self, filepath):
        self.tuning_records_file = filepath

    @property
    def pass_config(self):
        base = {"tir.disable_vectorize": True}
        extra = self.config["extra_pass_config"]
        if isinstance(extra, str):
            import ast

            extra = ast.literal_eval(extra)
        assert isinstance(extra, dict)
        base.update(extra)
        return base

    @property
    def target_device(self):
        return self.config["target_device"]

    @property
    def extra_target(self):
        return self.config["extra_target"]

    @property
    def opt_level(self):
        return self.config["opt_level"]

    @property
    def use_tuning_results(self):
        return bool(self.config["use_tuning_results"])

    @property
    def tvmc_extra_args(self):
        return self.config["tvmc_extra_args"]

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

    def get_pass_config_tvmc_args(self):
        args = []
        for key, value in self.pass_config.items():
            args.extend(["--pass-config", f"{key}={value}"])
        return args

    def get_disabled_pass_tvmc_args(self):
        args = []
        for item in self.config["disabled_passes"]:
            args.extend(["--disable-pass", item])
        return args

    def get_input_shapes_tvmc_args(self):
        if self.input_shapes is None:
            return []
        arg = " ".join([f"{name}:[" + ",".join(list(map(str, dims))) + "]" for name, dims in self.input_shapes.items()])
        return ["--input-shapes", arg]

    def get_common_tvmc_args(self, target="c"):
        if self.extra_target:
            # TODO: support multiple ones, currently only single one...
            target = ",".join([self.extra_target, target])
        return [
            str(self.model),
            "--target",
            target,
            *(["--target-c-device", self.target_device] if self.target_device is not None else []),
        ]

    def get_tuning_records_tvmc_args(self, target="c"):
        return (
            [
                "--tuning-records",
                str(self.tuning_records_file),
            ]
            if self.use_tuning_results and self.tuning_records_file is not None
            else []
        )

    def get_tvmc_compile_args(self, executor, fmt="mlf", target="c", runtime="crt"):
        assert executor in ["aot", "graph"], "Unsupported TVM executor"
        args = self.get_common_tvmc_args(target=target)
        args.extend(
            [
                "-f",
                fmt,
                "--executor",
                executor,
                "--runtime",
                runtime,
                *self.get_pass_config_tvmc_args(),
                *self.get_disabled_pass_tvmc_args(),
                "--opt-level",
                str(self.opt_level),
                *self.get_input_shapes_tvmc_args(),
                *self.get_tuning_records_tvmc_args(),
                *self.tvmc_extra_args,
                # TODO: also set --model-format? (optional)
            ]
        )
        return args

    def prepare_python_environment(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.config["tvm.pythonpath"])
        env["TVM_LIBRARY_PATH"] = str(self.config["tvm.build_dir"])
        return env

    def invoke_tvmc(self, command, *args, verbose=False):
        # print("invoke_tvmc", command, args, verbose)
        env = self.prepare_python_environment()
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        utils.python(*pre, command, *args, live=verbose, env=env)

    def invoke_tvmc_compile(self, out, dump=None, verbose=False):
        args = self.get_tvmc_compile_args()
        args.extend(["--output", str(out)])
        if dump:
            assert isinstance(dump, list)
            args.extend(["--dump-code", ",".join(dump)])
        self.invoke_tvmc("compile", *args, verbose=verbose)

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     assert isinstance(cls.name, str)
    #     cls.registry[cls.name] = cls

    def load_model(self, model):
        self.model = model
        with open(model, "rb") as handle:
            model_buf = handle.read()
            self.model_info = get_tflite_model_info(model_buf)
            self.input_shapes = {tensor.name: tensor.shape for tensor in self.model_info.inTensors}
