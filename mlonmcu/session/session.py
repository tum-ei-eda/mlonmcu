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
import multiprocessing
from time import sleep
from datetime import datetime
import pandas as pd

from enum import Enum
from datetime import datetime
import tempfile
from pathlib import Path
import os
import logging
import concurrent.futures
from tqdm import tqdm
from .run import RunStage


from mlonmcu.session.run import Run
from mlonmcu.logging import get_logger
from mlonmcu.report import Report

from .postprocess.postprocess import SessionPostprocess

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


class SessionStatus(Enum):
    CREATED = 0
    OPEN = 1
    CLOSED = 2
    ERROR = 3


class Session:
    def __init__(self, label="unnamed", idx=None, archived=False, dir=None):
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.label = label + "_" + timestamp
        self.idx = idx
        self.status = SessionStatus.CREATED
        self.opened_at = None
        self.closed_at = None
        self.runs = []
        self.next_run_idx = 0
        self.archived = archived
        if dir is None:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        else:
            self.tempdir = None
            self.dir = dir
            if not self.dir.is_dir():
                self.dir.mkdir(parents=True)
        self.runs_dir = self.dir / "runs"
        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)
        if not self.archived:
            self.open()

    @property
    def prefix(self):
        return f"[session-{self.idx}] " if self.idx else ""

    def create_run(self, *args, **kwargs):
        idx = len(self.runs)
        logger.debug("Creating a new run with id %s", idx)
        run = Run(*args, idx=idx, session=self, **kwargs)
        self.runs.append(run)
        # TODO: move this to a helper function
        run_link = run.dir.parent / "latest"  # TODO: Create relative symlink using os.path.relpath for portability
        if os.path.islink(run_link):
            os.unlink(run_link)
        os.symlink(run.dir, run_link)
        return run

    #  def update_run(self): # TODO TODO
    #      pass

    def get_reports(self):
        reports = [run.get_report() for run in self.runs]
        merged = Report()
        merged.add(reports)
        return merged

    def enumerate_runs(self):
        # Find start index
        max_idx = -1
        for run in self.runs:
            if run.archived:
                max_idx = max(max_idx, run.idx)
        run_idx = max_idx + 1
        for run in self.runs:
            if not run.archived:
                run.idx = run_idx
                run._init_directory()
                run_idx += 1
        self.next_run_idx = run_idx

    def request_run_idx(self):
        ret = self.next_run_idx
        self.next_run_idx += 1
        # TODO: find a better approach for this
        return ret

    def process_runs(
        self,
        until=RunStage.DONE,
        per_stage=False,
        num_workers=1,
        progress=False,
        export=False,
        context=None,
    ):

        # TODO: Add configurable callbacks for stage/run complete

        self.enumerate_runs()
        assert num_workers > 0, "num_workers can not be < 1"
        workers = []
        results = []
        workers = []
        pbar = None
        pbar2 = None
        num_runs = len(self.runs)
        num_failures = 0
        stage_failures = {}
        worker_run_idx = []

        def _init_progress(total, msg="Processing..."):
            global pbar
            pbar = tqdm(
                total=total,
                desc=msg,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
                leave=None,
            )

        def _update_progress(count=1):
            global pbar
            pbar.update(count)

        def _close_progress():
            global pbar
            if pbar:
                pbar.close()

        def _init_progress2(total, msg="Processing..."):
            global pbar2
            pbar2 = tqdm(
                total=total,
                desc=msg,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
            )

        def _update_progress2(count=1):
            global pbar2
            pbar2.update(count)

        def _close_progress2():
            global pbar2
            if pbar2:
                pbar2.close()

        def _process(run, until, skip):
            run.process(until=until, skip=skip, export=export, context=context)
            if progress:
                _update_progress()

        def _join_workers(workers):
            nonlocal num_failures
            results = []
            for i, w in enumerate(workers):
                try:
                    results.append(w.result())
                except Exception as e:
                    logger.exception(e)
                    logger.error("An exception was thrown by a worker during simulation")
                run_index = worker_run_idx[i]
                run = self.runs[run_index]
                if run.failing:
                    num_failures += 1
                    failed_stage = RunStage(run.next_stage).name
                    if failed_stage in stage_failures:
                        stage_failures[failed_stage].append(run_index)
                    else:
                        stage_failures[failed_stage] = [run_index]
            if progress:
                _close_progress()
            return results

        def _used_stages(runs, until):
            used = []
            for stage_index in list(range(RunStage.LOAD, until + 1)) + [RunStage.POSTPROCESS]:
                stage = RunStage(stage_index)
                if any(run.has_stage(stage) for run in runs):
                    used.append(stage)
            return used

        used_stages = _used_stages(self.runs, until)
        skipped_stages = [stage for stage in RunStage if stage not in used_stages]

        with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            if per_stage:
                if progress:
                    _init_progress2(len(used_stages), msg=f"Processing stages")
                for stage in used_stages:
                    run_stage = RunStage(stage).name
                    if progress:
                        _init_progress(len(self.runs), msg=f"Processing stage {run_stage}")
                    else:
                        logger.info(self.prefix + f"Processing stage {run_stage}")
                    for i, run in enumerate(self.runs):
                        if i == 0:
                            total_threads = min(len(self.runs), num_workers)
                            cpu_count = multiprocessing.cpu_count()
                            if (stage >= RunStage.COMPILE) and run.platform:
                                total_threads *= run.platform.num_threads
                            if total_threads > 2 * cpu_count:
                                if pbar2:
                                    print()
                                logger.warning(
                                    f"The chosen configuration leads to a maximum of {total_threads} threads being processed which heavily exceeds the available CPU resources ({cpu_count}). It is recommended to lower the value of 'mlif.num_threads'!"
                                )
                        if run.failing:
                            logger.warning(f"Skiping stage '{run_stage}' for failed run")
                        else:
                            worker_run_idx.append(i)
                            workers.append(executor.submit(_process, run, until=stage, skip=skipped_stages))
                    results = _join_workers(workers)
                    workers = []
                    worker_run_idx = []
                    if progress:
                        _update_progress2()
                if progress:
                    _close_progress2()
            else:
                if progress:
                    _init_progress(len(self.runs), msg="Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for i, run in enumerate(self.runs):
                    if i == 0:
                        total_threads = min(len(self.runs), num_workers)
                        cpu_count = multiprocessing.cpu_count()
                        if (until >= RunStage.COMPILE) and run.platform.name == "mlif":
                            total_threads *= (
                                run.platform.num_threads
                            )  # TODO: This should also be used for non-mlif platforms
                        if total_threads > 2 * cpu_count:
                            if pbar2:
                                print()
                            logger.warning(
                                f"The chosen configuration leads to a maximum of {total_threads} being processed which heavily exceeds the available CPU resources (cpu_count). It is recommended to lower the value of 'mlif.num_threads'!"
                            )
                    worker_run_idx.append(i)
                    workers.append(executor.submit(_process, run, until=until, skip=skipped_stages))
                results = _join_workers(workers)
        if num_failures == 0:
            logger.info("All runs completed successfuly!")
        elif num_failures == 0:
            logger.error("All runs have failed to complete!")
        else:
            num_success = num_runs - num_failures
            logger.warning(f"{num_success} out or {num_runs} runs completed successfully!")
            summary = "\n".join(
                [
                    f"\t{stage}: \t{len(failed)} failed run(s): " + " ".join([str(idx) for idx in failed])
                    for stage, failed in stage_failures.items()
                    if len(failed) > 0
                ]
            )
            logger.info("Summary:\n" + summary)

        report = self.get_reports()
        logger.info("Postprocessing session report")
        # Warning: currently we only support one instance of the same type of postprocess, also it will be applied to all rows!
        session_postprocesses = []
        for run in self.runs:
            for postprocess in run.postprocesses:
                if isinstance(postprocess, SessionPostprocess):
                    if postprocess.name not in [p.name for p in session_postprocesses]:
                        session_postprocesses.append(postprocess)
        for postprocess in session_postprocesses:
            artifacts = postprocess.post_session(report)
            if artifacts is not None:
                for artifact in artifacts:
                    # Postprocess has an artifact: write to disk!
                    logger.debug("Writting postprocess artifact to disk: %s", artifact.name)
                    artifact.export(self.dir)
        report_file = Path(self.dir) / "report.csv"
        report.export(report_file)
        results_dir = context.environment.paths["results"].path
        results_file = results_dir / f"{self.label}.csv"
        report.export(results_file)
        logger.info(self.prefix + "Done processing runs")
        print_report = True
        if print_report:
            logger.info("Report:\n" + str(report.df))

    def __repr__(self):
        return f"Session(idx={self.idx},status={self.status},runs={self.runs})"

    @property
    def active(self):
        return self.status == SessionStatus.OPEN

    def open(self):
        self.status = SessionStatus.OPEN
        self.opened_at = datetime.now()

    def close(self, err=None):
        if err:
            self.status = SessionStatus.ERROR
        else:
            self.status = SessionStatus.CLOSED
        self.closed_at = datetime.now()


# TODO: implement close()? and use closing contextlib? for tempdir
