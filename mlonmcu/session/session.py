from random import randint
from time import sleep
from datetime import datetime
import pandas as pd

from enum import Enum
from datetime import datetime
import tempfile
from pathlib import Path
import os
import logging
import concurrent
from tqdm import tqdm
from .run import RunStage


from mlonmcu.session.run import Run
from mlonmcu.logging import get_logger
from mlonmcu.report import Report

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


class SessionStatus(Enum):
    CREATED = 0
    OPEN = 1
    CLOSED = 2
    ERROR = 3


class Session:
    def __init__(self, alias="unnamed", idx=None, archived=False, dir=None):
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.alias = alias + "_" + timestamp
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
        run_link = run.dir.parent / "latest"
        if os.path.islink(run_link):
            os.unlink(run_link)
        os.symlink(run.dir, run_link)
        return run

    #  def update_run(self): # TODO TODO
    #      pass

    def get_reports(self):
        report_dfs = [run.get_report().df for run in self.runs]
        merged = Report()
        merged.df = pd.concat(report_dfs)
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

        def _process(run, until):
            run.process(until=until, export=export, context=context)
            # sleep(randint(1, 5))
            if progress:
                _update_progress()

        def _join_workers(workers):
            results = []
            for w in workers:
                try:
                    results.append(w.result())
                except Exception as e:
                    logger.exception(e)
                    logger.error(
                        "An exception was thrown by a worker during simulation"
                    )
            if progress:
                _close_progress()
            return results

        with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            if per_stage:
                if progress:
                    _init_progress2(int(until + 1), msg=f"Processing stages")
                for stage in range(until + 1):
                    run_stage = RunStage(stage).name
                    if progress:
                        _init_progress(
                            len(self.runs), msg=f"Processing stage {run_stage}"
                        )
                    else:
                        logger.info(self.prefix + f"Processing stage {run_stage}")
                    for run in self.runs:
                        workers.append(executor.submit(_process, run, until=stage))
                    results = _join_workers(workers)
                    workers = []
                    if progress:
                        _update_progress2()
                if progress:
                    _close_progress2()
            else:
                if progress:
                    _init_progress(len(self.runs), msg="Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for run in self.runs:
                    workers.append(executor.submit(_process, run, until=until))
                results = _join_workers(workers)
        report = self.get_reports()
        report_file = Path(self.dir) / "report.csv"
        report.export(report_file)
        results_dir = context.environment.paths["results"].path
        results_file = results_dir / f"{self.alias}.csv"
        report.export(results_file)
        logger.info(self.prefix + "Done processing runs")

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
