#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
"""Definition of MLonMCU session schedulers."""
import os
import shutil
import filelock
import tempfile
import multiprocessing
from datetime import datetime
from enum import Enum
from pathlib import Path
import concurrent.futures
from typing import List, Optional

from tqdm import tqdm

from mlonmcu.session.run import Run
from mlonmcu.logging import get_logger
from mlonmcu.report import Report
from mlonmcu.config import filter_config

from .postprocess.postprocess import SessionPostprocess
from .run import RunStage
from .progress import init_progress, update_progress, close_progress

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


def handle_executor(name: str):
    # TODO: handle (thread_pool, process_pool, remote, hybrid)
    EXECUTOR_LOOKUP = {
        "thread_pool": concurrent.futures.ThreadPoolExecutor,
    }
    ret = EXECUTOR_LOOKUP.get(name, None)
    assert ret is not None, f"Executor not found: {name}"
    return ret


def _process(pbar, run, until, skip, export):
    """Helper function to invoke the run."""
    run.process(until=until, skip=skip, export=export)


# TODO: alternative _process functions


def _used_stages(runs, until):
    """Determines the stages which are used by at least one run."""
    used = []
    for stage_index in list(range(RunStage.LOAD, until + 1)) + [RunStage.POSTPROCESS]:
        stage = RunStage(stage_index)
        if any(run.has_stage(stage) for run in runs):
            used.append(stage)
    return used


class SessionScheduler:
    """TODO"""

    def __init__(
        self,
        runs: List[Run],
        until: RunStage = RunStage.DONE,
        per_stage: bool = False,
        progress: bool = False,
        executor: str = "thread_pool",
        num_workers: int = 1,
    ):
        self.runs = runs
        self.until = until
        self.per_stage = per_stage
        self.progress = progress
        self._executor_cls = handle_executor(executor)
        self._executor_args = [num_workers]
        self.num_workers = num_workers
        self._futures = []
        # TODO: contextmanager?
        self.num_failures = 0
        self.stage_failures = {}
        # worker_run_idx = []
        self._future_run_idx = {}
        self.used_stages, self.skipped_stages = self.prepare()

    def prepare(self):
        used_stages = _used_stages(self.runs, self.until)
        skipped_stages = [stage for stage in RunStage if stage not in used_stages]
        return used_stages, skipped_stages

    @property
    def num_runs(self):
        return len(self.runs)

    @property
    def num_success(self):
        return self.num_runs - self.num_failures

    def reset(self):
        raise NotImplementedError(".reset() not implemented")

    def _join_futures(self, pbar):
        """Helper function to collect all worker threads."""
        results = []
        for f in concurrent.futures.as_completed(self._futures):
            try:
                results.append(f.result())
            except Exception as e:
                logger.exception(e)
                logger.error("An exception was thrown by a worker during simulation")
            if self.progress:
                update_progress(pbar)
            run_index = self._future_run_idx[f]
            run = self.runs[run_index]
            if run.failing:
                self.num_failures += 1
                failed_stage = RunStage(run.next_stage).name
                if failed_stage in self.stage_failures:
                    self.stage_failures[failed_stage].append(run_index)
                else:
                    self.stage_failures[failed_stage] = [run_index]
        if self.progress:
            close_progress(pbar)
        return results

    def process(
        self,
        export=False,
        context=None,
    ):
        pbar = None  # Outer progress bar
        pbar2 = None  # Inner progress bar

        with self._executor_cls(*self._executor_args) as executor:
            if self.per_stage:
                if self.progress:
                    pbar2 = init_progress(len(self.used_stages), msg="Processing stages")
                for stage in self.used_stages:
                    run_stage = RunStage(stage).name
                    if self.progress:
                        pbar = init_progress(len(self.runs), msg=f"Processing stage {run_stage}")
                    else:
                        logger.info("%s Processing stage %s", self.prefix, run_stage)
                    for i, run in enumerate(self.runs):
                        if i == 0:
                            total_threads = min(self.num_runs, self.num_workers)
                            cpu_count = multiprocessing.cpu_count()
                            if (stage == RunStage.COMPILE) and run.compile_platform:
                                total_threads *= run.compile_platform.num_threads
                            if total_threads > 2 * cpu_count:
                                if pbar2:
                                    print()
                                logger.warning(
                                    "The chosen configuration leads to a maximum of %d threads being"
                                    + " processed which heavily exceeds the available CPU resources (%d)."
                                    + " It is recommended to lower the value of 'mlif.num_threads'!",
                                    total_threads,
                                    cpu_count,
                                )
                        if run.failing:
                            logger.warning("Skiping stage '%s' for failed run", run_stage)
                        else:
                            f = executor.submit(
                                _process, pbar, run, until=stage, skip=self.skipped_stages, export=export
                            )
                            self._futures.append(f)
                            self._future_run_idx[f] = i
                    self._join_futures(pbar)
                    self._futures = []
                    self._future_run_idx = {}
                    if self.progress:
                        update_progress(pbar2)
                if self.progress:
                    close_progress(pbar2)
            else:
                if self.progress:
                    pbar = init_progress(self.num_runs, msg="Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for i, run in enumerate(self.runs):
                    if i == 0:
                        total_threads = min(len(self.runs), self.num_workers)
                        cpu_count = multiprocessing.cpu_count()
                        if (
                            (self.until >= RunStage.COMPILE)
                            and run.compile_platform is not None
                            and run.compile_platform.name == "mlif"
                        ):
                            total_threads *= (
                                run.compile_platform.num_threads
                            )  # TODO: This should also be used for non-mlif platforms
                        if total_threads > 2 * cpu_count:
                            if pbar2:
                                print()
                            logger.warning(
                                "The chosen configuration leads to a maximum of %d being processed which"
                                + " heavily exceeds the available CPU resources (%d)."
                                + " It is recommended to lower the value of 'mlif.num_threads'!",
                                total_threads,
                                cpu_count,
                            )
                    f = executor.submit(_process, pbar, run, until=self.until, skip=self.skipped_stages)
                    self._futures.append(f)
                    self._future_run_idx[f] = i
                self._join_futures(pbar)
        return self.runs
        # return num_failures == 0

    def postprocess(self, report, dest):
        logger.info("Postprocessing session report")
        # Warning: currently we only support one instance of the same type of postprocess,
        # also it will be applied to all rows!
        session_postprocesses = []
        for run in self.runs:
            for postprocess in run.postprocesses:
                if isinstance(postprocess, SessionPostprocess):
                    if postprocess.name not in [p.name for p in session_postprocesses]:
                        session_postprocesses.append(postprocess)
        if self.progress:
            pbar = init_progress(len(session_postprocesses), msg="Postprocessing session")
        for postprocess in session_postprocesses:
            try:
                artifacts = postprocess.post_session(report)
            except Exception as e:
                logger.exception(e)
                self.num_failing += 1
                break
            if self.progress:
                update_progress(pbar)
            if artifacts is not None:
                for artifact in artifacts:
                    # Postprocess has an artifact: write to disk!
                    logger.debug("Writing postprocess artifact to disk: %s", artifact.name)
                    artifact.export(dest)
        if self.progress:
            close_progress(pbar)
        return report

    def print_summary(self):
        if self.num_failures == 0:
            logger.info("All runs completed successfuly!")
        elif self.num_failures == self.num_runs:
            logger.error("All runs have failed to complete!")
        else:
            logger.warning("%d out or %d runs completed successfully!", self.num_success, self.num_runs)
            summary = "\n".join(
                [
                    f"\t{stage}: \t{len(failed)} failed run(s): " + " ".join([str(idx) for idx in failed])
                    for stage, failed in self.stage_failures.items()
                    if len(failed) > 0
                ]
            )
            logger.info("Summary:\n%s", summary)
