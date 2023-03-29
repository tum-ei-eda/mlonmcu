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
"""TVM Platform"""

import re
import time
import tempfile
import concurrent
from pathlib import Path

from mlonmcu.setup import utils
from mlonmcu.logging import get_logger
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.target.metrics import Metrics
from mlonmcu.target import get_targets
from mlonmcu.target.target import Target
from mlonmcu.config import str2bool
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_bench_tvmc_args,
    get_data_tvmc_args,
    get_rpc_tvmc_args,
    get_target_tvmc_args,
)
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.flow.tvm.backend.tuner import get_autotuning_defaults, get_autotvm_defaults, get_autoscheduler_defaults

from ..platform import TargetPlatform, BuildPlatform, TunePlatform
from .tvm_target import create_tvm_platform_target
from .tvm_backend import create_tvm_platform_backend, get_tvm_platform_backends

logger = get_logger()


class TvmPlatform(BuildPlatform, TargetPlatform, TunePlatform):
    """TVM Platform class."""

    FEATURES = TargetPlatform.FEATURES + [
        "benchmark",
        "tvm_rpc",
        "autotvm",
        "autoschedule",
        "tvm_profile",
    ]  # TODO: validate?

    DEFAULTS = {
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "fill_mode": "random",
        "ins_file": None,
        "outs_file": None,
        "print_top": False,
        "profile": False,
        "repeat": 1,
        "number": 1,
        "aggregate": "none",  # Allowed: avg, max, min, none, all
        "total_time": False,
        "use_rpc": False,
        "rpc_key": None,
        "rpc_hostname": None,
        "rpc_port": None,
        "tvmc_extra_args": [],
        "tvmc_custom_script": None,
        "experimental_tvmc_tune_tasks": False,
        "experimental_tvmc_tune_visualize": False,
        **{("autotuning_" + key): value for key, value in get_autotuning_defaults().items()},
        **{("autotvm_" + key): value for key, value in get_autotvm_defaults().items()},
        **{("autoscheduler_" + key): value for key, value in get_autoscheduler_defaults().items()},
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(
            "tvm",  # Actually: tvmllvm
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    def create_backend(self, name):
        supported = self.get_supported_backends()
        assert name in supported, f"{name} is not a valid TVM platform backend"
        base = supported[name]
        return create_tvm_platform_backend(name, self, base=base)

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
    def number(self):
        return self.config["number"]

    @property
    def aggregate(self):
        value = self.config["aggregate"]
        assert value in ["avg", "all", "max", "min", "none"]
        return value

    @property
    def total_time(self):
        value = self.config["total_time"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
    def experimental_tvmc_tune_tasks(self):
        value = self.config["experimental_tvmc_tune_tasks"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def experimental_tvmc_tune_visualize(self):
        value = self.config["experimental_tvmc_tune_visualize"]
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
        backend_names = get_tvm_platform_backends()
        return backend_names

    def get_supported_targets(self):
        # TODO: get this via tvmc run --help
        target_names = ["cpu", "cuda", "cl", "metal", "vulkan", "rocm", "micro"]

        skip_names = ["micro"]  # Use microtvm platform instead

        return [f"tvm_{name}" for name in target_names if name not in skip_names]

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid TVM device"
        targets = get_targets()
        if name in targets:
            base = targets[name]
        else:
            base = Target
        return create_tvm_platform_target(name, self, base=base)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_tvmc_run_args(self, path, device):
        return [
            path,
            *["--device", device],
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                print_time=True, profile=self.profile, end_to_end=False, repeat=self.repeat, number=self.number
            ),
            *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
        ]

    def invoke_tvmc(self, command, *args):
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=self.print_outputs, print_output=False, env=env)

    def invoke_tvmc_run(self, path, device):
        args = self.get_tvmc_run_args(path, device)
        return self.invoke_tvmc("run", *args)

    def run(self, elf, target, timeout=120):
        # TODO: implement timeout
        # Here, elf is actually a directory
        # TODO: replace workaround with possibility to pass TAR directly
        tar_path = elf
        output = self.invoke_tvmc_run(str(tar_path), target.device)

        return output

    def get_tune_args(self, model, backend, target, out):
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
                backend.target,
                extra_target=backend.extra_target,
                target_details=backend.get_target_details(),
            ),
            *(["--desired-layout", desired_layout] if desired_layout is not None else []),
            *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
            # TODO: missing: pass config, disabled_pass, etc.
            *(["--early-stopping", str(early_stopping)] if early_stopping > 0 else []),
            *["--parallel", str(max_parallel)],
            *["--timeout", str(timeout * max_parallel)],
            *["--trials", str(trials)],
            *["--number", str(self.number)],  # TODO: increase while tuning?
            *["--repeat", str(self.repeat)],  # TODO: increase while tuning?
            *(["--tuning-records", results_file] if results_file is not None else []),
            *["--output", str(out)],
        ]
        autotvm_enable = self.config["autotvm_enable"]
        autoscheduler_enable = self.config["autoscheduler_enable"]
        if autotvm_enable:
            tuner = self.config.get("autotvm_tuner", "ga")
            assert tuner in ["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"]
            ret.extend(["--tuner", tuner])
            if self.config["autotuning_visualize"]:
                assert (
                    self.experimental_tvmc_tune_tasks
                ), f"{self.name}.visualize_tuning requires experimental_autotvm_visualize"
                ret.append("--visualize")
        elif autoscheduler_enable:
            ret.append("--enable-autoscheduler")
            if self.config.get("autoscheduler_include_simple_tasks", False):
                ret.append("--include-simple-tasks")
            if self.config.get("autoscheduler_log_estimated_latency", False):
                ret.append("--log-estimated-latency")
            hardware_details = target.get_hardware_details()
            if len(hardware_details) > 0:
                for key, value in hardware_details.items():
                    ret.extend([f"--{key}", str(value)])
        if self.config["autotuning_tasks"]:
            assert self.experimental_tvmc_tune_tasks, f"{self.name}.tune_tasks requires experimental_tvmc_tune_tasks"
            ret.extend(["--tasks", str(self.config["autotuning_tasks"])])
        ret.append(model)
        return ret

    def _tune_model(self, model_path, backend, target):
        autotvm_enable = self.config["autotvm_enable"]
        autoscheduler_enable = self.config["autoscheduler_enable"]
        assert [autotvm_enable, autoscheduler_enable].count(
            True
        ) == 1, "Can not use AutoTVM and AutoScheduler at the same time"
        results_file = self.config["autotuning_results_file"]
        append = self.config["autotuning_append"]
        num_workers = self.config["autotuning_num_workers"]
        artifacts = []
        verbose = False
        if self.print_outputs:
            verbose = True

        def remove_empty(inp):
            return [line for line in inp if len(line.strip()) > 0]

        def count_failed_trials(inp):
            cnt = 0
            for line in inp.split("\n"):
                m = re.compile(r".*\[1000000000\.0\].*").match(line)
                if m:
                    cnt += 1
            return cnt

        # pick best records
        def _pick_best(backend, records, verbose=False):
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
                env = prepare_python_environment(backend.tvm_pythonpath, backend.tvm_build_dir, backend.tvm_configs_dir)
                utils.python("-m", "tvm.autotvm.record", *args, live=verbose, env=env)
                with open(out_file, "r") as handle:
                    content_best = handle.read()
            return content_best

        content = ""
        total_size = None
        if autotvm_enable or autoscheduler_enable:
            if append:
                if results_file is not None:
                    with open(results_file, "r") as handle:
                        content = handle.read()

            sub_metrics = {}
            if num_workers is not None:
                if isinstance(num_workers, str):
                    num_workers = int(num_workers)
                assert isinstance(num_workers, int) and num_workers > 0
                assert self.experimental_tvmc_tune_tasks, "num_workers requires experimental_tvmc_tune_tasks=1"
                # TODO: fix
                assert self.config["autotuning_tasks"] is None, "tune_tasks not supported together with num_workers > 1"

                def get_tune_tasks():
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        out_file = Path(tmp_dir) / "tuning_results.log.txt"
                        tune_args = self.get_tune_args(model_path, backend, target, out_file)
                        out = self.invoke_tvmc("tune", *tune_args, "--tasks", "list")
                        lines = out.split("\n")
                        for i, line in enumerate(lines):
                            if "Available Tasks for tuning" in line:
                                lines = lines[i + 1 :]
                                break
                        # tasks = [line.split(". ", 1)[1] for line in lines if len(line.strip()) > 0]
                        # Get config space sizes
                        matches = re.compile(r"(\d+). Task.*\(len=(\d+)\)").findall(out)
                        sizes = list(map(lambda x: (int(x[0]), int(x[1])), matches))
                        return sizes

                # num_tasks = len(get_tune_tasks())
                tune_tasks = get_tune_tasks()
                workers = []
                with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
                    # for i in range(num_tasks):
                    for i, task_len in tune_tasks:
                        if total_size is None:
                            total_size = 0
                        total_size += task_len
                        print(f"Created worker for task {i}")

                        def do_work(idx, prepend, task_len):
                            t0 = time.time()
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                out_file = Path(tmp_dir) / "tuning_results.log.txt"
                                with open(out_file, "w") as handle:
                                    handle.write(prepend)
                                # TODO: divide trials by number of tasks?
                                tune_args = self.get_tune_args(model_path, backend, target, out_file)
                                out = self.invoke_tvmc("tune", *tune_args, "--tasks", str(idx))
                                with open(out_file, "r") as handle:
                                    content = handle.read()
                                # content_best = _pick_best(backend, content, verbose=verbose)
                                sub_trials = len(remove_empty(content.split("\n")))
                                sub_failed_trials = count_failed_trials(content)
                                t1 = time.time()
                            return (out, content, task_len, sub_trials, sub_failed_trials, t1 - t0)

                        workers.append(executor.submit(do_work, i, content, task_len))
                all_out = ""
                all_content = ""
                for i, w in enumerate(workers):
                    print(f"Worker {i}: done")
                    ret = w.result()
                    out, content, size, tuned, failed, duration = ret
                    all_out += out
                    all_content += content
                    metrics_ = Metrics()
                    metrics_.add("Config Space Size", size, True)
                    metrics_.add("Total Trials", tuned, True)
                    metrics_.add("Failed Trials", failed, True)
                    metrics_.add("Tune Duration [s]", duration, True)
                    metrics_.add("Tune Duration per Trial [s]", duration / tuned + failed, True)
                    trials = self.config.get("autotuning_trials", 10)
                    if not isinstance(trials, int):
                        trials = int(trials)
                    early_stopping = self.config.get("autotuning_early_stopping", None)
                    if not isinstance(early_stopping, int):
                        early_stopping = int(early_stopping)
                    if early_stopping is None:
                        early_stopping = max(trials, 10)  # Let's see if this default works out...
                    if early_stopping < trials:
                        early = tuned + failed < min(trials, size)
                    else:
                        early = False
                    metrics_.add("Early Stopped", early, True)
                    sub_metrics[f"task{i}"] = metrics_
                out = all_out
                content = all_content
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_file = Path(tmp_dir) / "tuning_results.log.txt"
                    with open(out_file, "w") as handle:
                        handle.write(content)
                    tune_args = self.get_tune_args(model_path, backend, target, out_file)
                    out = self.invoke_tvmc("tune", *tune_args)
                    with open(out_file, "r") as handle:
                        content = handle.read()
        else:
            if results_file is None:
                return {}, {}
            assert Path(results_file).is_file()
            with open(results_file, "r") as handle:
                content = handle.read()

        artifact = Artifact("tuning_results.log.txt", content=content, fmt=ArtifactFormat.TEXT, flags=["records"])
        artifacts.append(artifact)

        metrics = Metrics()

        if total_size is not None:
            metrics.add("Config Space Size", total_size, True)

        content_best = _pick_best(backend, content, verbose=verbose)
        total_trials = len(remove_empty(content.split("\n")))
        metrics.add("Total Trials", total_trials, True)

        failed_trials = count_failed_trials(content)
        metrics.add("Failed Trials", failed_trials, True)
        if len(content_best) > 0:
            artifact_ = Artifact("best_tuning_results.log.txt", content=content_best, fmt=ArtifactFormat.TEXT)
            artifacts.append(artifact_)
            num_tuned = len(remove_empty(content_best.split("\n")))
            metrics.add("Tuned Tasks", num_tuned, True)
        else:
            metrics.add("Tuned Tasks", 0, True)

        if autotvm_enable or autoscheduler_enable:
            stdout_artifact = Artifact(
                "tvmc_tune_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmaot_out.log?
            artifacts.append(stdout_artifact)

        return {"default": artifacts}, {"default": metrics, **sub_metrics}
