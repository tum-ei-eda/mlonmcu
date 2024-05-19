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
import shutil
import filelock
import tempfile
from typing import Optional, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from mlonmcu.session.run import Run, RunInitializer, RunResult
from mlonmcu.logging import get_logger
from mlonmcu.report import Report
from mlonmcu.config import filter_config
from mlonmcu.config import str2bool

from .run import RunStage
from .schedule import SessionScheduler

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
        # "process_pool": False,
        "executor": "thread_pool",
        "use_init_stage": False,
        "shuffle": False,
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
        self.dir = dir
        self.tempdir = None
        self.session_lock = None

    @property
    def runs_dir(self):
        return None if self.dir is None else (self.dir / "runs")

    def __enter__(self):
        if self.archived:
            logger.warning("Opening an already archived session is not recommended")
        else:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            self.close()

    @property
    def prefix(self):
        """get prefix property."""
        return f"[session-{self.idx}] " if self.idx else ""

    @property
    def report_fmt(self):
        """get report_fmt property."""
        return str(self.config["report_fmt"])

    # @property
    # def process_pool(self):
    #     """get process_pool property."""
    #     value = self.config["process_pool"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def executor(self):
        """get executor property."""
        return str(self.config["executor"])

    @property
    def use_init_stage(self):
        """get use_init_stage property."""
        value = self.config["use_init_stage"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def shuffle(self):
        """get shuffle property."""
        value = self.config["shuffle"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def needs_initializer(self):
        """TODO"""
        return self.executor == "process_pool" or self.use_init_stage

    def create_run(self, *args, **kwargs):
        """Factory method to create a run and add it to this session."""
        idx = len(self.runs)
        logger.debug("Creating a new run with id %s", idx)
        if self.needs_initializer:
            run = RunInitializer(*args, idx=idx, **kwargs)
        else:
            run = Run(*args, idx=idx, **kwargs)
        self.runs.append(run)
        return run

    def add_run(self, run: Union[Run, RunInitializer], ignore_idx: bool = True):
        """TODO."""
        if ignore_idx:
            idx = len(self.runs)
        else:
            raise NotImplementedError
        if isinstance(run, RunInitializer):
            self.runs.append(run)
        elif isinstance(run, Run):
            raise NotImplementedError
        else:
            assert False
        logger.debug("Importing run with id %s", idx)

    #  def update_run(self): # TODO TODO
    #      pass

    def get_reports(self, results: Optional[RunResult] = None):
        """Returns a full report which includes all runs in this session."""
        if self.report:
            return self.report

        if results is None:
            logger.warning("Use of session.get_reports without args is deprecated. Please pass list of RunResults!")

            reports = [run.get_report(session=self) for run in self.runs if not isinstance(run, RunInitializer)]
        else:
            reports = [res.get_report(session=self) for res in results]

        merged = Report()
        merged.add(reports)
        return merged

    def enumerate_runs(self):
        """Update run indices."""
        # Find start index
        max_idx = -1
        for run in self.runs:
            if not isinstance(run, RunInitializer) and run.archived:
                max_idx = max(max_idx, run.idx)
        run_idx = max_idx + 1
        last_run_idx = None
        for run in self.runs:
            if isinstance(run, RunInitializer):
                run.idx = run_idx
                run_idx += 1
                last_run_idx = run.idx
            elif not run.archived:
                run.idx = run_idx
                # run.init_directory(session=self)
                run.init_directory(parent=self.runs_dir)
                run_idx += 1
                last_run_idx = run.idx
        self.next_run_idx = run_idx
        if last_run_idx is not None:
            self.update_latest_run_symlink(last_run_idx)

    def update_latest_run_symlink(self, latest_run_idx):
        run_link = self.runs_dir / "latest"  # TODO: Create relative symlink using os.path.relpath for portability
        if os.path.islink(run_link):
            os.unlink(run_link)
        os.symlink(self.runs_dir / str(latest_run_idx), run_link)

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
        noop=False,
    ):
        """Process a runs in this session until a given stage."""

        # TODO: Add configurable callbacks for stage/run complete
        assert self.active, "Session needs to be opened first"

        assert len(self.runs), "List of runs is empty"
        self.enumerate_runs()
        self.report = None
        assert num_workers > 0, "num_workers can not be < 1"
        scheduler = SessionScheduler(
            self.runs,
            until,
            executor=self.executor,
            per_stage=per_stage,
            progress=progress,
            num_workers=num_workers,
            use_init_stage=self.use_init_stage,
            session=self,
            shuffle=self.shuffle,
        )
        if noop:
            logger.info(self.prefix + "Skipping processing of runs")
            scheduler.initialize(context=context)
            return 0

        self.runs, results = scheduler.process(export=export, context=context)
        report = self.get_reports(results=results)
        scheduler.print_summary()
        report = scheduler.postprocess(report, dest=self.dir)
        report_file = Path(self.dir) / f"report.{self.report_fmt}"
        report.export(report_file)
        results_dir = context.environment.paths["results"].path
        results_file = results_dir / f"{self.label}.{self.report_fmt}"
        report.export(results_file)
        logger.info(self.prefix + "Done processing runs")
        self.report = report
        if print_report:
            logger.info("Report:\n%s", str(report.df))

        return scheduler.num_failures == 0

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

    @property
    def failing(self):
        """Get failng property."""

        # via report
        if self.report:
            df = self.report.df
            if "Failing" in df.columns:
                if df["Failing"].any():
                    return True
        # via runs
        if len(self.runs) > 0:
            for run in self.runs:
                if run.failing:
                    return True

        return False

    def open(self):
        """Open this run."""
        self.status = SessionStatus.OPEN
        self.opened_at = datetime.now()
        if dir is None:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        else:
            if not self.dir.is_dir():
                self.dir.mkdir(parents=True)
        self.session_lock = filelock.FileLock(os.path.join(self.dir, ".lock"))
        try:
            self.session_lock.acquire(timeout=10)
        except filelock.Timeout as err:
            raise RuntimeError("Lock on session could not be aquired.") from err
        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)

    def close(self, err=None):
        """Close this run."""
        if err:
            self.status = SessionStatus.ERROR
        else:
            self.status = SessionStatus.CLOSED
        self.closed_at = datetime.now()
        self.session_lock.release()
        os.remove(self.session_lock.lock_file)
        if self.tempdir:
            self.tempdir.cleanup()
