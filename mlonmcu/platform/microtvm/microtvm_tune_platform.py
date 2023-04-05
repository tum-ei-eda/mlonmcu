import re
from mlonmcu.config import str2bool
import time
import tempfile
import concurrent
from pathlib import Path
from .microtvm_target_platform import MicroTvmTargetPlatform
from ..platform import TunePlatform

from mlonmcu.flow.tvm.backend.tuner import get_autotuning_defaults, get_autotvm_defaults, get_autoscheduler_defaults
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_rpc_tvmc_args,
    get_target_tvmc_args,
)
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.target.metrics import Metrics
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.setup import utils


class MicroTvmTunePlatform(TunePlatform, MicroTvmTargetPlatform):
    """MicroTVM Tune platform class."""

    FEATURES = TunePlatform.FEATURES + MicroTvmTargetPlatform.FEATURES + ["autotvm", "autoschedule"]

    DEFAULTS = {
        **TunePlatform.DEFAULTS,
        **MicroTvmTargetPlatform.DEFAULTS,
        "experimental_tvmc_tune_tasks": False,
        "experimental_tvmc_tune_visualize": False,
        "min_repeat_ms": 0,
        **{("autotuning_" + key): value for key, value in get_autotuning_defaults().items()},
        **{("autotvm_" + key): value for key, value in get_autotvm_defaults().items()},
        **{("autoscheduler_" + key): value for key, value in get_autoscheduler_defaults().items()},
    }

    REQUIRED = TunePlatform.REQUIRED + MicroTvmTargetPlatform.REQUIRED + []

    @property
    def experimental_tvmc_micro_tune(self):
        value = self.config["experimental_tvmc_micro_tune"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def experimental_tvmc_print_time(self):
        value = self.config["experimental_tvmc_print_time"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def experimental_tvmc_tune_tasks(self):
        value = self.config["experimental_tvmc_tune_tasks"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def min_repeat_ms(self):
        value = self.config["min_repeat_ms"]
        return int(value)

    def invoke_tvmc_micro_tune(self, *args, target=None, list_options=False):
        all_args = []
        all_args.extend(args)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("tune", *all_args, target=target, list_options=list_options)

    def invoke_tvmc_tune(self, *args, target=None):
        return self.invoke_tvmc_micro_tune(*args, target=target)

    def get_tune_args_base(self, model, backend, target, out):
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
            *["--min-repeat-ms", str(self.min_repeat_ms)],
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

    def get_tune_args(self, model, backend, target, out):
        return self.get_tune_args_base(model, backend, target, out)

    def _tune_model_base(self, model_path, backend, target):
        print("_tm self.config", self.config)
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
                        out = self.invoke_tvmc_tune(*tune_args, "--tasks", "list", target=target)
                        lines = out.split("\n")
                        for i, line in enumerate(lines):
                            if "Available Tasks for tuning" in line:
                                lines = lines[i + 1:]
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
                                out = self.invoke_tvmc_tune(*tune_args, "--tasks", str(idx), target=target)
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
                    out = self.invoke_tvmc_tune(*tune_args, target=target)
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

    def _tune_model(self, model_path, backend, target):
        assert self.experimental_tvmc_micro_tune, "Microtvm tuning requires experimental_tvmc_micro_tune"
        self._tune_model_base(model_path, backend, target)
