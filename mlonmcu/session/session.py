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
"""Definition of a MLonMCU Run which represents a set of benchmarks in a session."""
import os
import time
import shutil
import tempfile
import multiprocessing
from datetime import datetime
from enum import Enum
from pathlib import Path
import concurrent.futures

from tqdm import tqdm

from mlonmcu.session.run import Run
from mlonmcu.logging import get_logger
from mlonmcu.report import Report
from mlonmcu.config import filter_config

from .postprocess.postprocess import SessionPostprocess
from .run import RunStage

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


class SessionStatus(Enum):  # TODO: remove?
    """Status type for a session."""

    CREATED = 0
    OPEN = 1
    CLOSED = 2
    ERROR = 3


class Session:
    """A session which wraps around multiple runs in a context."""

    DEFAULTS = {
        "report_fmt": "csv",
    }

    def __init__(self, label="", idx=None, archived=False, dir=None, config=None):
        self.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.label = (
            label if len(label) > 0 else ("unnamed" + "_" + self.timestamp)
        )  # TODO: decide if named sessions should also get a timestamp?
        self.idx = idx
        self.config = config if config else {}
        self.config = filter_config(self.config, "session", self.DEFAULTS, set(), set())
        self.status = SessionStatus.CREATED
        self.opened_at = None
        self.closed_at = None
        self.runs = []
        self.report = None
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
        """get prefix property."""
        return f"[session-{self.idx}] " if self.idx else ""

    @property
    def report_fmt(self):
        """get report_fmt property."""
        return str(self.config["report_fmt"])

    def create_run(self, *args, **kwargs):
        """Factory method to create a run and add it to this session."""
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
        """Returns a full report which includes all runs in this session."""
        if self.report:
            return self.report

        reports = [run.get_report() for run in self.runs]
        merged = Report()
        merged.add(reports)
        return merged

    def enumerate_runs(self):
        """Update run indices."""
        # Find start index
        max_idx = -1
        for run in self.runs:
            if run.archived:
                max_idx = max(max_idx, run.idx)
        run_idx = max_idx + 1
        for run in self.runs:
            if not run.archived:
                run.idx = run_idx
                run.init_directory()
                run_idx += 1
        self.next_run_idx = run_idx

    def request_run_idx(self):
        """Return next free run index."""
        ret = self.next_run_idx
        self.next_run_idx += 1
        # TODO: find a better approach for this
        return ret

    def process_runs(
        self,
        until=RunStage.DONE,
        per_stage=False,
        print_report=False,
        num_workers=1,
        progress=False,
        export=False,
        context=None,
    ):
        """Process a runs in this session until a given stage."""

        # TODO: Add configurable callbacks for stage/run complete

        self.enumerate_runs()
        self.report = None
        assert num_workers > 0, "num_workers can not be < 1"
        workers = []
        # results = []
        workers = []
        pbar = None  # Outer progress bar
        pbar2 = None  # Inner progress bar
        # use_wandb = False
        use_wandb = True
        wandb_ = None
        runs_pending = 0
        runs_running = 0
        runs_done = 0
        runs_failing = 0
        stage_times = []
        num_runs = len(self.runs)
        num_failures = 0
        stage_failures = {}
        worker_run_idx = []

        def _init_progress(total, msg="Processing..."):
            """Helper function to initialize a progress bar for the session."""
            return tqdm(
                total=total,
                desc=msg,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
                leave=None,
            )
        def _init_wandb():
            import wandb

            # start a new wandb run to track this script
            wandb_run_global = wandb.init(
                # set the wandb project where this run will be logged
                project="mlonmcu",
                group=f"session_{self.idx}",
                # name="?",
                job_type="global",
                # tags=[],
                # notes=""
                config={
                    "foo": "bar",
                },
                reinit=True,
            )
            # wandb.log({"acc": acc, "loss": loss})
            return wandb_run_global
        def _init_wandb_run(run):
            import wandb

            # start a new wandb run to track this script
            wandb_run_local = wandb.init(
                # set the wandb project where this run will be logged
                project="mlonmcu",
                group=f"session_{self.idx}",
                name=f"run_{run.idx}",
                job_type="local",
                # tags=[],
                # notes=""
                config={
                    "foo2": "bar2",
                },
                reinit=True,
            )
            # wandb.log({"acc": acc, "loss": loss})
            return wandb_run_local


        def _update_progress(pbar, count=1):
            """Helper function to update the progress bar for the session."""
            pbar.update(count)

        def _reset_wandb(wandb_, stage, pending=0):
            nonlocal runs_pending, runs_running, runs_done, runs_failing, stage_times
            runs_pending = pending
            runs_done = 0
            runs_running = 0
            runs_failing = 0
            stage_times = []
            wandb_.log(
                {
                    "session.runs.stage": stage,
                    "session.runs.pending": runs_pending,
                    "session.runs.running": runs_running,
                    "session.runs.failing": runs_failing,
                    "session.runs.done": runs_done,
                    "session.runs.stage_time.min": 0,
                    "session.runs.stage_time.max": 0,
                    "session.runs.stage_time.mean": 0,
                }
            )

        def _update_wandb_pending(wandb_, count=1):
            nonlocal runs_pending
            runs_pending += count
            wandb_.log(
                {
                    "session.runs.pending": runs_pending,
                }
            )


        def _update_wandb_running(wandb_, count=1, t=0):
            nonlocal runs_running
            runs_running += count
            wandb_.log(
                {
                    "session.runs.pending": runs_pending,
                }
            )

        def _update_wandb_done(wandb_, count=1, t=0):
            nonlocal runs_pending, runs_running, runs_done, stage_times
            runs_pending -= count
            runs_running -= count
            runs_done += count
            stage_times.append(t)
            wandb_.log(
                {
                    "session.runs.pending": runs_pending,
                    "session.runs.running": runs_running,
                    "session.runs.done": runs_done,
                    "session.runs.stage_time.min": min(stage_times),
                    "session.runs.stage_time.max": max(stage_times),
                    "session.runs.stage_time.mean": sum(stage_times) / len(stage_times),
                }
            )

        def _update_wandb_failing(wandb_, count=1):
            nonlocal runs_failing
            runs_failing += count
            wandb_.log(
                {
                    "session.runs.failing": runs_failing,
                }
            )

        def _close_progress(pbar):
            """Helper function to close the session progressbar, if available."""
            if pbar:
                pbar.close()

        def _close_wandb(wandb_):
            """Helper function to close the session progressbar, if available."""
            wandb_.finish()

        def _process(pbar, wandb_, run, until, skip):
            """Helper function to invoke the run."""
            if use_wandb:
                _update_wandb_running(wandb_)
            t0 = time.time()
            run.process(until=until, skip=skip, export=export)
            t1 = time.time()
            if use_wandb:
                _update_wandb_done(wandb_, t=t1-t0)
            if progress:
                _update_progress(pbar)

        def _join_workers(workers):
            """Helper function to collect all worker threads."""
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
                    if use_wandb:
                        _update_wandb_failing(wandb_)
                    failed_stage = RunStage(run.next_stage).name
                    if failed_stage in stage_failures:
                        stage_failures[failed_stage].append(run_index)
                    else:
                        stage_failures[failed_stage] = [run_index]
            if progress:
                _close_progress(pbar)
            return results

        def _used_stages(runs, until):
            """Determines the stages which are used by at least one run."""
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
                    pbar2 = _init_progress(len(used_stages), msg="Processing stages")
                if use_wandb:
                    wandb_ = _init_wandb()
                for stage in used_stages:
                    if use_wandb:
                        _reset_wandb(wandb_, stage)
                    run_stage = RunStage(stage).name
                    if progress:
                        pbar = _init_progress(len(self.runs), msg=f"Processing stage {run_stage}")
                    else:
                        logger.info("%s Processing stage %s", self.prefix, run_stage)
                    for i, run in enumerate(self.runs):
                        if i == 0:
                            total_threads = min(len(self.runs), num_workers)
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
                            worker_run_idx.append(i)
                            workers.append(executor.submit(_process, pbar, wandb_, run, until=stage, skip=skipped_stages))
                            if use_wandb:
                                _update_wandb_pending(wandb_)
                    _join_workers(workers)
                    workers = []
                    worker_run_idx = []
                    if progress:
                        _update_progress(pbar2)
                if progress:
                    _close_progress(pbar2)
            else:
                if progress:
                    pbar = _init_progress(len(self.runs), msg="Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for i, run in enumerate(self.runs):
                    if i == 0:
                        total_threads = min(len(self.runs), num_workers)
                        cpu_count = multiprocessing.cpu_count()
                        if (
                            (until >= RunStage.COMPILE)
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
                    worker_run_idx.append(i)
                    workers.append(executor.submit(_process, pbar, run, until=until, skip=skipped_stages))
                _join_workers(workers)
        if num_failures == 0:
            logger.info("All runs completed successfuly!")
        elif num_failures == num_runs:
            logger.error("All runs have failed to complete!")
        else:
            num_success = num_runs - num_failures
            logger.warning("%d out or %d runs completed successfully!", num_success, num_runs)
            summary = "\n".join(
                [
                    f"\t{stage}: \t{len(failed)} failed run(s): " + " ".join([str(idx) for idx in failed])
                    for stage, failed in stage_failures.items()
                    if len(failed) > 0
                ]
            )
            logger.info("Summary:\n%s", summary)

        report = self.get_reports()
        logger.info("Postprocessing session report")
        # Warning: currently we only support one instance of the same type of postprocess,
        # also it will be applied to all rows!
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
        report_file = Path(self.dir) / f"report.{self.report_fmt}"
        report.export(report_file)
        if use_wandb:
            wandb_.save(str(report_file))
            _close_wandb(wandb_)
            for run in self.runs:
                wandb2_ = _init_wandb_run(run)
                if run.report is not None:
                    wandb2_.log(run.report.df.to_dict())
                _close_wandb(wandb2_)
        results_dir = context.environment.paths["results"].path
        results_file = results_dir / f"{self.label}.{self.report_fmt}"
        report.export(results_file)
        logger.info(self.prefix + "Done processing runs")
        self.report = report
        if print_report:
            logger.info("Report:\n%s", str(report.df))

        return num_failures == 0

    def discard(self):
        """Discard a run and remove its directory."""
        self.close()
        if self.dir.is_dir():
            logger.debug("Cleaning up discarded session")
            shutil.rmtree(self.dir)

    def __repr__(self):
        return f"Session(idx={self.idx},status={self.status},runs={self.runs})"

    @property
    def active(self):
        """Get active property."""
        return self.status == SessionStatus.OPEN

    def open(self):
        """Open this run."""
        self.status = SessionStatus.OPEN
        self.opened_at = datetime.now()

    def close(self, err=None):
        """Close this run."""
        if err:
            self.status = SessionStatus.ERROR
        else:
            self.status = SessionStatus.CLOSED
        self.closed_at = datetime.now()
        if self.tempdir:
            self.tempdir.cleanup()


# TODO: implement close()? and use closing contextlib? for tempdir
