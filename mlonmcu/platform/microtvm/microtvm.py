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
"""MicroTVM Platform"""

import re
import tempfile
import concurrent
from pathlib import Path
from typing import Tuple

from mlonmcu.setup import utils
from mlonmcu.config import str2bool
from mlonmcu.logging import get_logger
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_bench_tvmc_args,
    get_data_tvmc_args,
    get_target_tvmc_args,
    get_rpc_tvmc_args,
)
from mlonmcu.flow.tvm.backend.tuner import TVMTuner

from ..platform import CompilePlatform, TargetPlatform, BuildPlatform, TunePlatform
from .microtvm_target import create_microtvm_platform_target, get_microtvm_platform_targets
from .microtvm_backend import create_microtvm_platform_backend, get_microtvm_platform_backends

logger = get_logger()
# TODO: This file is very similar to the TVM platform -> Reuse as much as possible


def parse_project_options_from_stdout(out):
    return re.compile(r"^\s+([A-Za-z0-9_]+)=", re.MULTILINE).findall(out)


def filter_project_options(valid, options):
    return {key: value for key, value in options.items() if key in valid}


def get_project_option_args(template, stage, project_options):
    ret = []
    for key, value in project_options.items():
        ret.append(f"{key}={value}")

    if len(ret) > 0:
        ret = ["--project-option"] + ret

    return ret


class MicroTvmPlatform(CompilePlatform, TargetPlatform, BuildPlatform, TunePlatform):
    """TVM Platform class."""

    FEATURES = (
        CompilePlatform.FEATURES + TargetPlatform.FEATURES + ["autotune", "tvm_rpc", "tvm_profile"]
    )  # TODO: validate?

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_options": {},
        "project_dir": None,
        "fill_mode": "random",
        "ins_file": None,
        "outs_file": None,
        "print_top": False,
        "profile": False,
        "repeat": 1,
        "use_rpc": False,
        "rpc_key": None,
        "rpc_hostname": None,
        "rpc_port": None,
        "tvmc_extra_args": [],
        "tvmc_custom_script": None,
        "experimental_tvmc_micro_tune": False,
        "experimental_tvmc_tune_tasks": False,
        "experimental_autotvm_visualize": False,
        "experimental_tvmc_print_time": False,
        **{("autotuning_" + key): value for key, value in TVMTuner.DEFAULTS.items()},
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(
            "microtvm",  # Actually: tvmllvm
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    @property
    def project_options(self):
        opts = self.config["project_options"]
        if isinstance(opts, str):
            opts_split = opts.split(" ")
            opts = {}
            for opt in opts_split:
                key, value = opt.split("=")[:2]
                opts[key] = value
        assert isinstance(opts, dict)
        return opts

    @property
    def fill_mode(self):
        return self.config["fill_mode"]

    @property
    def ins_file(self):
        return self.config["ins_file"]

    @property
    def outs_file(self):
        return self.config["outs_file"]

    @property
    def print_top(self):
        return self.config["print_top"]

    @property
    def profile(self):
        value = self.config["profile"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def repeat(self):
        return self.config["repeat"]

    @property
    def use_rpc(self):
        value = self.config["use_rpc"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def rpc_key(self):
        return self.config["rpc_key"]

    @property
    def rpc_hostname(self):
        return self.config["rpc_hostname"]

    @property
    def rpc_port(self):
        return self.config["rpc_port"]

    @property
    def tvmc_extra_args(self):
        return self.config["tvmc_extra_args"]

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

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
    def project_template(self):
        return self.config["project_template"]

    @property
    def visualize_tuning(self):
        # Visualize the tuning progress via matplotlib
        value = self.config["visualize_tuning"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def tune_tasks(self):
        # Effectively select which tasks should be tuned in the session
        return self.config["tune_tasks"]

    @property
    def experimental_tvmc_micro_tune(self):
        value = self.config["experimental_tvmc_micro_tune"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def experimental_tvmc_tune_tasks(self):
        value = self.config["experimental_tvmc_tune_tasks"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def experimental_tvmc_tune_visualize(self):
        value = self.config["experimental_tvmc_tune_visualize"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def experimental_tvmc_print_time(self):
        value = self.config["experimental_tvmc_print_time"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return
        dir_name = self.name
        if path is not None:
            self.project_dir = Path(path)
        elif self.config["project_dir"] is not None:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.debug(
                    "Creating temporary directory because no context was available "
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    def get_supported_backends(self):
        backend_names = get_microtvm_platform_backends()
        return backend_names

    def create_backend(self, name):
        supported = self.get_supported_backends()
        assert name in supported, f"{name} is not a valid MicroTVM platform backend"
        base = supported[name]
        return create_microtvm_platform_backend(name, self, base=base)

    def get_supported_targets(self):
        return get_microtvm_platform_targets()

    def create_target(self, name):
        supported = self.get_supported_targets()
        assert name in supported, f"{name} is not a valid MicroTVM device"
        base = supported[name]
        return create_microtvm_platform_target(name, self, base=base)

    def get_tvmc_run_args(self, path, device, list_options=False):
        if self.use_rpc:
            raise RuntimeError("RPC is only supported for tuning with microtvm platform")
        if self.profile:
            assert (
                self.experimental_tvmc_print_time
            ), "MicroTVM profiloing is only supported in environments with microtvm.experimental_tvmc_print_time=1"
        ret = [
            path,
            *["--device", device],
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                print_time=self.experimental_tvmc_print_time and not self.profile,
                profile=self.profile and self.experimental_tvmc_print_time,
                end_to_end=False,
                # repeat=self.repeat if self.experimental_tvmc_print_time else None,
                # number=self.number if self.experimental_tvmc_print_time else None,
            ),
            # *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
        ]
        if list_options:
            ret.append("--list-options")
        return ret

    def get_tvmc_micro_args(self, command, path, mlf_path, template, list_options=False, tune_args=None):
        ret = [command]
        if "create" in command:
            ret.extend(["--force", path, mlf_path])
        elif command == "tune":
            if tune_args:
                ret.extend(tune_args)
            if list_options:
                ret.extend(["--output", "-", "_"])
        else:
            ret.append(path)
        ret.extend(template)
        if list_options:
            ret.append("--help")
        return ret

    def invoke_tvmc(self, command, *args, target=None, prefix=""):
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if target:
            target.update_environment(env)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=self.print_outputs, print_output=False, env=env, prefix=prefix)

    def collect_available_project_options(self, command, path, mlf_path, template, micro=True, target=None):
        args = self.get_tvmc_micro_args(command, path, mlf_path, template, list_options=True)
        out = self.invoke_tvmc("micro", *args, target=target)
        return parse_project_options_from_stdout(out)

    def invoke_tvmc_micro(
        self, command, path, mlf_path, template, target, extra_args=None, micro=True, tune_args=None, prefix=""
    ):
        args = self.get_tvmc_micro_args(command, path, mlf_path, template, tune_args=tune_args)
        options = filter_project_options(
            self.collect_available_project_options(command, path, mlf_path, template, target=target),
            target.get_project_options(),
        )
        args += get_project_option_args(template, command, options)
        return self.invoke_tvmc("micro", *args, target=target, prefix=prefix)

    def collect_available_run_project_options(self, path, device):
        args = self.get_tvmc_run_args(path, device, list_options=True)
        out = self.invoke_tvmc("run", *args)
        return parse_project_options_from_stdout(out)

    def invoke_tvmc_run(self, path, device, template, target, micro=True):
        args = self.get_tvmc_run_args(path, device)
        if micro:
            options = filter_project_options(
                self.collect_available_run_project_options(path, device), target.get_project_options()
            )
            args.extend(get_project_option_args(template, "run", options))
        return self.invoke_tvmc("run", *args, target=target)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_template_args(self, target):
        template = target.template
        if target.template_path:
            template = "template"
            template_path = target.template_path
        else:
            if template == "template":
                assert self.project_template is not None
                template_path = self.project_template
            else:
                template_path = None
        if template_path:
            return ("template", "--template-dir", template_path)
        else:
            return (template,)

    def prepare(self, mlf, target):
        out = self.invoke_tvmc_micro("create", self.project_dir, mlf, self.get_template_args(target), target)
        return out

    def compile(self, target):
        out = ""
        # TODO: build with cmake options
        out += self.invoke_tvmc_micro("build", self.project_dir, None, self.get_template_args(target), target)
        # TODO: support self.num_threads (e.g. patch esp-idf)
        return out

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        # TODO: name missleading as we are not interested in the ELF
        src = Path(src) / "default.tar"  # TODO: lookup for *.tar file
        artifacts = []
        out = self.prepare(src, target)
        out += self.compile(target)
        stdout_artifact = Artifact(
            "microtvm_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {}

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for microtvm platform")
        # TODO: implement timeout
        logger.debug("Flashing target software using MicroTVM ProjectAPI")
        output = self.invoke_tvmc_micro("flash", self.project_dir, None, self.get_template_args(target), target)
        return output

    def run(self, elf, target, timeout=120):
        # TODO: implement timeout
        output = self.flash(elf, target)
        output += self.invoke_tvmc_run(
            str(self.project_dir), "micro", self.get_template_args(target), target, micro=True
        )
        return output

    def get_micro_tune_args(self, model, backend, out):
        tuner = self.config.get("autotuning_tuner", "ga")
        assert tuner in ["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"]
        trials = self.config.get("autotuning_trials", 10)
        if not isinstance(trials, int):
            trials = int(trials)
        early_stopping = self.config.get("autotuning_early_stopping", None)
        if early_stopping is None:
            early_stopping = max(trials, 10)  # Let's see if this default works out...
        early_stopping = int(early_stopping)
        max_parallel = int(self.config.get("autotuning_max_parallel", 1))
        timeout = int(self.config.get("autotuning_timeout", 1000))
        results_file = self.config.get("autotuning_results_file", None)
        desired_layout = backend.config.get("desired_layout", None)
        ret = [
            *get_target_tvmc_args(
                "c",
                extra_target=backend.extra_target,
                target_details=backend.get_target_details(),
            ),
            *(["--desired-layout", desired_layout] if desired_layout is not None else []),
            *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
            # TODO: missing: pass config, disabled_pass, etc.
            *["--tuner", tuner],
            *(["--early-stopping", str(early_stopping)] if early_stopping > 0 else []),
            *["--parallel", str(max_parallel)],
            *["--timeout", str(timeout * max_parallel)],
            *["--trials", str(trials)],
            *["--number", str(1)],  # TODO: variable
            *["--repeat", str(1)],  # TODO: variable
            *(["--tuning-records", results_file] if results_file is not None else []),
            *["--output", str(out)],
            # "--target-c-link-params",
            # "1",
        ]
        if self.config["autotuning_visualize"]:
            assert (
                self.experimental_tvmc_tune_tasks
            ), f"{self.name}.visualize_tuning requires experimental_autotvm_visualize"
            ret.append("--visualize")
        if self.config["autotuning_tasks"]:
            assert self.experimental_tvmc_tune_tasks, f"{self.name}.tune_tasks requires experimental_tvmc_tune_tasks"
            ret.extend(["--tasks", str(self.config["autotuning_tasks"])])
        ret.append(model)
        return ret

    def _tune_model(self, model_path, backend, target):
        assert self.experimental_tvmc_micro_tune, "Microtvm tuning requires experimental_tvmc_micro_tune"
        enable = self.config["autotuning_enable"]
        results_file = self.config["autotuning_results_file"]
        append = self.config["autotuning_append"]
        num_workers = int(self.config["autotuning_num_workers"])
        artifacts = []
        verbose = False
        if self.print_outputs:
            verbose = True

        content = ""
        if enable:
            if append:
                if results_file is not None:
                    with open(results_file, "r") as handle:
                        content = handle.read()

            if num_workers > 1:
                assert self.experimental_tvmc_tune_tasks, "num_workers>1 requires experimental_tvmc_tune_tasks=1"
                # TODO: fix
                assert self.config["autotuning_tasks"] is None, "tune_tasks not supported together with num_workers > 1"

                def get_tune_tasks():
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        out_file = Path(tmp_dir) / "tuning_results.log.txt"
                        tune_args = self.get_micro_tune_args(model_path, backend, out_file)
                        out = self.invoke_tvmc_micro(
                            "tune",
                            self.project_dir,
                            None,
                            self.get_template_args(target),
                            target,
                            tune_args=tune_args + ["--task", "list"],
                        )
                        lines = out.split("\n")
                        for i, line in enumerate(lines):
                            if "Available Tasks for tuning" in line:
                                lines = lines[i + 1 :]
                                break
                        tasks = [line.split(". ", 1)[1] for line in lines if len(line.strip()) > 0]
                        return tasks

                num_tasks = len(get_tune_tasks())
                workers = []
                with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
                    for i in range(num_tasks):
                        print(f"Created worker for task {i}")

                        def do_work(idx, prepend):
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                out_file = Path(tmp_dir) / "tuning_results.log.txt"
                                with open(out_file, "w") as handle:
                                    handle.write(prepend)
                                tune_args = self.get_micro_tune_args(model_path, backend, out_file)
                                out = self.invoke_tvmc_micro(
                                    "tune",
                                    self.project_dir,
                                    None,
                                    self.get_template_args(target),
                                    target,
                                    tune_args=tune_args + ["--task", str(idx)],
                                    prefix=f"[worker-{idx}] ",
                                )
                                with open(out_file, "r") as handle:
                                    content = handle.read()
                            return (out, content)

                        workers.append(executor.submit(do_work, i, content))
                all_out = ""
                all_content = ""
                for i, w in enumerate(workers):
                    print(f"Worker {i}: done")
                    ret = w.result()
                    out, content = ret
                    all_out += out
                    all_content += content
                out = all_out
                content = all_content
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_file = Path(tmp_dir) / "tuning_results.log.txt"
                    with open(out_file, "w") as handle:
                        handle.write(content)
                    tune_args = self.get_micro_tune_args(model_path, backend, out_file)
                    out = self.invoke_tvmc_micro(
                        "tune", self.project_dir, None, self.get_template_args(target), target, tune_args=tune_args
                    )
                    with open(out_file, "r") as handle:
                        content = handle.read()

        artifact = Artifact("tuning_results.log.txt", content=content, fmt=ArtifactFormat.TEXT)
        artifacts.append(artifact)

        # pick best records
        def _pick_best(backend, records, verbose=False):
            content_best = ""
            with tempfile.TemporaryDirectory() as tmp_dir:
                in_file = Path(tmp_dir) / "tuning_results.log.txt"
                with open(in_file, "w") as handle:
                    handle.write(records)
                out_file = Path(tmp_dir) / "best_tuning_results.log.txt"
                args = [
                    "--mode",
                    "pick",
                    "--i",
                    in_file,
                    "--o",
                    out_file,
                ]
                env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
                utils.python("-m", "tvm.autotvm.record", *args, live=verbose, env=env)
                with open(out_file, "r") as handle:
                    content_best = handle.read()
            return content_best

        content_best = _pick_best(backend, content, verbose=verbose)
        if len(content_best) > 0:
            artifact_ = Artifact("best_tuning_results.log.txt", content=content_best, fmt=ArtifactFormat.TEXT)
            artifacts.append(artifact_)

        if enable:
            stdout_artifact = Artifact(
                "tvmc_micro_tune_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmaot_out.log?
            artifacts.append(stdout_artifact)

        return {"default": artifacts}, {}
