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
"""TVM Tune Platform"""
import re
import os
from mlonmcu.config import str2bool
import time
import tempfile
import tarfile
import concurrent
from pathlib import Path
from .tvm_target_platform import TvmTargetPlatform
from ..platform import TunePlatform

from mlonmcu.flow.tvm.backend.tuner import (
    get_autotuning_defaults,
    get_autotvm_defaults,
    get_autoscheduler_defaults,
    get_metascheduler_defaults,
)
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_rpc_tvmc_args,
    get_target_tvmc_args,
    get_disabled_pass_tvmc_args,
)
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.target.metrics import Metrics
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

logger = get_logger()


class TvmTunePlatform(TunePlatform, TvmTargetPlatform):
    """TVM Tune platform class."""

    FEATURES = TunePlatform.FEATURES | TvmTargetPlatform.FEATURES | {"autotvm", "autoscheduler", "metascheduler"}

    DEFAULTS = {
        **TunePlatform.DEFAULTS,
        **TvmTargetPlatform.DEFAULTS,
        "experimental_tvmc_tune_tasks": False,
        "experimental_tvmc_tune_visualize": False,
        # "experimental_tvmc_tune_wandb": False,  # TODO
        "enable_wandb": False,
        "min_repeat_ms": 0,
        **{("autotuning_" + key): value for key, value in get_autotuning_defaults().items()},
        **{("autotvm_" + key): value for key, value in get_autotvm_defaults().items()},
        **{("autoscheduler_" + key): value for key, value in get_autoscheduler_defaults().items()},
        **{("metascheduler_" + key): value for key, value in get_metascheduler_defaults().items()},
    }

    REQUIRED = TunePlatform.REQUIRED | TvmTargetPlatform.REQUIRED

    @property
    def tune_tasks(self):
        # Effectively select which tasks should be tuned in the session
        return self.config["tune_tasks"]

    @property
    def experimental_tvmc_tune_tasks(self):
        value = self.config["experimental_tvmc_tune_tasks"]
        return str2bool(value)

    @property
    def experimental_tvmc_tune_visualize(self):
        value = self.config["experimental_tvmc_tune_visualize"]
        return str2bool(value)

    @property
    def enable_wandb(self):
        value = self.config["enable_wandb"]
        return str2bool(value)

    @property
    def min_repeat_ms(self):
        value = self.config["min_repeat_ms"]
        return int(value)

    def invoke_tvmc_tune(self, *args, target=None, **kwargs):
        return self.invoke_tvmc("tune", *args, target=target, **kwargs)

    def get_tune_args(self, model, backend, target, out, trials, early_stopping):
        max_parallel = int(self.config.get("autotuning_max_parallel", 1))
        timeout = int(self.config.get("autotuning_timeout", 1000))
        results_file = self.config.get("autotuning_results_file", None)
        desired_layout = backend.config.get("desired_layout", None)
        ret = [
            *get_target_tvmc_args(
                backend.target,
                extra_targets=backend.extra_targets,
                target_details=backend.get_target_details(),
                extra_target_details=backend.extra_target_details,
            ),
            *(["--desired-layout", desired_layout] if desired_layout is not None else []),
            *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
            *get_disabled_pass_tvmc_args(backend.disabled_passes),
            # TODO: missing: pass config etc.
            *(["--early-stopping", str(early_stopping)] if early_stopping > 0 else []),
            *["--parallel", str(max_parallel)],
            *["--timeout", str(timeout * max_parallel)],
            *["--trials", str(trials)],
            *["--number", str(self.number)],  # TODO: increase while tuning?
            *["--repeat", str(self.repeat)],  # TODO: increase while tuning?
            *["--min-repeat-ms", str(self.min_repeat_ms)],
            *(["--tuning-records", results_file] if results_file is not None else []),
            *["--output", str(out)],
        ]
        if self.config["autotuning_tasks"]:
            assert self.experimental_tvmc_tune_tasks, f"{self.name}.tune_tasks requires experimental_tvmc_tune_tasks"
            ret.extend(["--tasks", str(self.config["autotuning_tasks"])])
        if self.enable_wandb:
            ret.append("--wandb-callback")
        ret.append(model)
        return ret

    def get_autotvm_tune_args(self, model, backend, target, out, trials_global, early_stopping):
        ret = self.get_tune_args(model, backend, target, out, trials_global, early_stopping)

        tuner = self.config.get("autotvm_tuner", "ga")
        assert tuner in ["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"]
        ret.extend(["--tuner", tuner])
        if self.config["autotuning_visualize"]:
            to_file = self.config["autotuning_visualize_file"]
            if not to_file:
                to_file = "viz.png"
            live = self.config["autotuning_visualize_live"]
            assert self.experimental_tvmc_tune_tasks, "requires experimental_autotvm_visualize"
            visualize_arg = to_file
            if live:
                visualize_arg += ",live"
            ret.extend(["--visualize", visualize_arg])
        return ret

    def get_autoscheduler_tune_args(self, model, backend, target, out, trials_global, early_stopping):
        assert not self.enable_wandb, "WANDB callback not yet supported by AutoScheduler"
        ret = self.get_tune_args(model, backend, target, out, trials_global, early_stopping)
        ret.append("--enable-autoscheduler")
        if self.config.get("autoscheduler_include_simple_tasks", False):
            ret.append("--include-simple-tasks")
        if self.config.get("autoscheduler_log_estimated_latency", False):
            ret.append("--log-estimated-latency")
        hardware_details = target.get_hardware_details()
        if len(hardware_details) > 0:
            for key, value in hardware_details.items():
                ret.extend([f"--{key}", str(value)])
        return ret

    def get_metascheduler_tune_args(self, model, backend, target, out, trials_global, trials_single, early_stopping):
        ret = self.get_tune_args(model, backend, target, out, trials_global, early_stopping)
        ret.append("--enable-metascheduler")
        if trials_single:
            ret.extend(["--trials-per-task", str(trials_single)])
        return ret

    def _tune_model(self, model_path, backend, target):
        autotvm_enable = self.config["autotvm_enable"]
        autoscheduler_enable = self.config["autoscheduler_enable"]
        metascheduler_enable = self.config["metascheduler_enable"]
        if not autotvm_enable and not autoscheduler_enable and not metascheduler_enable:
            # Tuning not enabled! (Might happen if abstract autotune feature is used!)
            return {}, {}
        assert [autotvm_enable, autoscheduler_enable, metascheduler_enable].count(
            True
        ) == 1, "Can not use AutoTVM and AutoScheduler/MetaScheduler at the same time"
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

        def get_max_flops(out, prefix="M"):
            res = re.compile(r"\d+\.\d+\s*\/\s*(\d+\.\d+)\s+{prefix}FLOPS").findall(out)
            if len(res) > 0:
                return res[-1]
            return -1

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
        visualize_raw = None
        if num_workers is not None:
            if isinstance(num_workers, str):
                num_workers = int(num_workers)
            assert isinstance(num_workers, int) and num_workers >= 0
        trials_global = self.config.get("autotuning_trials", 10)
        trials_single = self.config.get("autotuning_trials_single", None)
        if not isinstance(trials_global, int):
            trials_global = int(trials_global)
        if trials_single is not None and not isinstance(trials_single, int):
            trials_single = int(trials_single)
        trials = trials_single if trials_single is not None else trials_global
        assert isinstance(trials, int)
        early_stopping = self.config.get("autotuning_early_stopping", None)
        if early_stopping is None:
            early_stopping = max(trials, 10)  # Let's see if this default works out...
        if not isinstance(early_stopping, int):
            early_stopping = int(early_stopping)
        assert isinstance(early_stopping, int)
        if metascheduler_enable:
            assert not append, "append not supported by MetaScheduler"
            assert num_workers is None or int(num_workers) == 0, "num_workers > 0 not supported by MetaScheduler"
            assert not self.config["autotuning_visualize"], "autotuning_visualize not supported by MetaScheduler"

            sub_metrics = {}
            sub_artifacts = {}
            with tempfile.TemporaryDirectory() as tmp_dir:
                # out_file = Path(tmp_dir) / "tuning_results.log.txt"
                tmp_dir = Path(tmp_dir)
                work_dir = tmp_dir / "work_dir"
                tune_args = self.get_metascheduler_tune_args(
                    model_path, backend, target, work_dir, trials_global, trials_single, 0
                )
                out = self.invoke_tvmc_tune(*tune_args, target=target, cwd=tmp_dir)
                with tarfile.open(tmp_dir / "work_dir.tar", "w") as tar:
                    for file in os.listdir(work_dir):
                        tar.add(work_dir / file, arcname=os.path.join("work_dir", file))
                raw = None
                with open(tmp_dir / "work_dir.tar", "rb") as tar:
                    raw = tar.read()
                artifact = Artifact(
                    "work_dir.tar", raw=raw, fmt=ArtifactFormat.ARCHIVE, flags=["records", "metascheduler"]
                )
        elif autotvm_enable or autoscheduler_enable:
            if append:
                if results_file is not None:
                    with open(results_file, "r") as handle:
                        content = handle.read()

            sub_metrics = {}
            sub_artifacts = {}
            if num_workers is not None and num_workers > 0:
                assert self.experimental_tvmc_tune_tasks, "num_workers requires experimental_tvmc_tune_tasks=1"
                # TODO: fix
                assert self.config["autotuning_tasks"] is None, "tune_tasks not supported together with num_workers > 0"

                def get_tune_tasks():
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        out_file = Path(tmp_dir) / "tuning_results.log.txt"
                        if autotvm_enable:
                            tune_args = self.get_autotvm_tune_args(model_path, backend, target, out_file, 1, 0)
                        elif autoscheduler_enable:
                            tune_args = self.get_autoscheduler_tune_args(model_path, backend, target, out_file, 1, 0)
                        else:
                            assert False
                        out = self.invoke_tvmc_tune(
                            *tune_args, "--tasks", "list", target=target, cwd=tmp_dir, live=False
                        )
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
                        logger.debug(f"Created worker for task {i}")

                        def do_work(idx, prepend, task_len):
                            nonlocal trials_single, trials_global, early_stopping
                            t0 = time.time()
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                out_file = Path(tmp_dir) / "tuning_results.log.txt"
                                with open(out_file, "w") as handle:
                                    handle.write(prepend)
                                if trials_single == 0 or (
                                    trials_single is None
                                ):  # 0: auto, None: do not limit per task
                                    trials_single = max(1, trials_global // len(tune_tasks))
                                    early_stopping = max(trials_single, 10)  # Let's see if this default works out...
                                if autotvm_enable:
                                    tune_args = self.get_autotvm_tune_args(
                                        model_path, backend, target, out_file, trials_single, early_stopping
                                    )
                                elif autoscheduler_enable:
                                    tune_args = self.get_autoscheduler_tune_args(
                                        model_path, backend, target, out_file, trials_single, early_stopping
                                    )
                                else:
                                    assert False
                                out = self.invoke_tvmc_tune(*tune_args, "--tasks", str(idx), target=target, cwd=tmp_dir)
                                with open(out_file, "r") as handle:
                                    content = handle.read()
                                visualize_raw_task = None
                                if self.config["autotuning_visualize"]:
                                    to_file = self.config["autotuning_visualize_file"]
                                    if not to_file or to_file is True:
                                        to_file = Path(tmp_dir) / "viz.png"
                                    else:
                                        to_file = Path(to_file)
                                    assert to_file.is_file()
                                    with open(to_file, "rb") as handle:
                                        visualize_raw_task = handle.read()
                                # content_best = _pick_best(backend, content, verbose=verbose)
                                sub_trials = len(remove_empty(content.split("\n")))
                                sub_failed_trials = count_failed_trials(content)
                                max_flops = get_max_flops(out)
                                t1 = time.time()
                            return (
                                out,
                                content,
                                task_len,
                                sub_trials,
                                sub_failed_trials,
                                max_flops,
                                t1 - t0,
                                visualize_raw_task,
                            )

                        workers.append(executor.submit(do_work, i, content, task_len))
                all_out = ""
                all_content = ""
                for i, w in enumerate(workers):
                    logger.debug(f"Worker {i}: pending")
                    metrics_ = Metrics()
                    artifacts_ = []
                    try:
                        ret = w.result()
                        logger.debug(f"Worker {i}: done")
                        out, content, size, tuned, failed, max_flops, duration, visualize_raw_task = ret
                        all_out += out
                        all_content += content
                        metrics_.add("Config Space Size", size, True)
                        metrics_.add("Total Trials", tuned, True)
                        metrics_.add("Failed Trials", failed, True)
                        metrics_.add("Max. MFLOPS", max_flops, True)
                        metrics_.add("Tune Duration [s]", duration, True)
                        metrics_.add("Tune Duration per Trial [s]", duration / tuned + failed, True)
                        if early_stopping < trials_single:
                            early = tuned + failed < min(trials_single, size)
                        else:
                            early = False
                        metrics_.add("Early Stopped", early, True)
                        if visualize_raw_task:
                            visualize_artifact = Artifact(
                                f"tuning_progress_task{i}.png",
                                raw=visualize_raw_task,
                                fmt=ArtifactFormat.RAW,
                                flags=["visualize"],
                            )
                            artifacts_.append(visualize_artifact)
                    except AssertionError:
                        logger.exception(f"Worker {i}: failed")
                        metrics_.add("Failed Tuning", True)
                    sub_metrics[f"task{i}"] = metrics_
                    sub_artifacts[f"task{i}"] = artifacts_
                out = all_out
                content = all_content
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_file = Path(tmp_dir) / "tuning_results.log.txt"
                    with open(out_file, "w") as handle:
                        handle.write(content)
                    if autotvm_enable:
                        tune_args = self.get_autotvm_tune_args(
                            model_path, backend, target, out_file, trials_global, early_stopping
                        )
                    elif autoscheduler_enable:
                        tune_args = self.get_autoscheduler_tune_args(
                            model_path, backend, target, out_file, trials_global, early_stopping
                        )  # TODO: expose per_task trials
                    else:
                        assert False
                    out = self.invoke_tvmc_tune(*tune_args, target=target, cwd=tmp_dir)
                    with open(out_file, "r") as handle:
                        content = handle.read()
                    visualize_raw = None
                    if self.config["autotuning_visualize"]:
                        to_file = self.config["autotuning_visualize_file"]
                        if not to_file or to_file is True:
                            to_file = Path(tmp_dir) / "viz.png"
                        else:
                            to_file = Path(tmp_dir)
                        assert to_file.is_file()
                        with open(to_file, "rb") as handle:
                            visualize_raw = handle.read()
        else:
            if results_file is None:
                return {}, {}
            assert Path(results_file).is_file()
            with open(results_file, "r") as handle:
                content = handle.read()

        if metascheduler_enable:
            artifacts.append(artifact)
            # TODO: get num trials etc.
            metrics = Metrics()
        elif autotvm_enable or autoscheduler_enable:
            flag = "autotvm" if not autoscheduler_enable else "autoscheduler"
            artifact = Artifact(
                "tuning_results.log.txt", content=content, fmt=ArtifactFormat.TEXT, flags=["records", flag]
            )
            artifacts.append(artifact)
            if visualize_raw:
                visualize_artifact = Artifact(
                    "tuning_progress.png", raw=visualize_raw, fmt=ArtifactFormat.RAW, flags=["visualize"]
                )
                artifacts.append(visualize_artifact)

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
                artifact_ = Artifact("best_tuning_results.log.txt", content="", fmt=ArtifactFormat.TEXT)
                artifacts.append(artifact_)
                metrics.add("Tuned Tasks", 0, True)

        if autotvm_enable or autoscheduler_enable or metascheduler_enable:
            stdout_artifact = Artifact(
                "tvmc_tune_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmaot_out.log?
            artifacts.append(stdout_artifact)

        return {"default": artifacts, **sub_artifacts}, {"default": metrics, **sub_metrics}
