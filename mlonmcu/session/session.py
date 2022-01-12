from random import randint
from time import sleep

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

logger = logging.getLogger("mlonmcu")


class SessionStatus(Enum):
    CREATED = 0
    OPEN = 1
    CLOSED = 2
    ERROR = 3


class Session:
    def __init__(self, idx=None, archived=False, dir=None):
        self.idx = idx
        self.status = SessionStatus.CREATED
        self.opened_at = None
        self.closed_at = None
        self.runs = []
        self.archived = archived
        if dir is None:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        else:
            self.tempdir = None
            self.dir = dir
            if not self.dir.is_dir():
                os.mkdir(self.dir)
        self.runs_dir = self.dir / "runs"
        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)
        if not self.archived:
            self.open()

    #  def create_run(self, *args, **kwargs):
    #      idx = self.run_id + 1
    #      logger.debug("Creating a new run with id %s", idx)
    #      run = Run(*args, idx=idx, session=self, **kwargs)
    #      self.runs.append(run)
    #      self.run_id = idx
    #      return run
    #      # TODO: set latest symlink?

    #  def update_run(self): # TODO TODO
    #      pass

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
                run_idx += 1

    def process_runs(
        self,
        until=RunStage.DONE,
        per_stage=False,
        num_workers=1,
        progress=False,
        context=None,
    ):
        assert num_workers > 0, "num_workers can not be < 1"
        workers = []
        results = []
        workers = []
        pbar = None

        def _init_progress(total, msg="Processing..."):
            global pbar
            pbar = tqdm(
                total=total,
                desc=msg,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )

        def _update_progress(count=1):
            global pbar
            pbar.update(count)

        def _close_progress():
            global pbar
            if pbar:
                pbar.close()

        def _process(run, until):
            run.process(until=until, context=context)
            sleep(randint(1, 5))
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
                for stage in range(until + 1):
                    run_stage = RunStage(stage).name
                    if progress:
                        _init_progress(
                            len(self.runs), msg=f"Processing stage {run_stage}"
                        )
                    else:
                        print(f"Processing stage {run_stage}")
                    for run in self.runs:
                        workers.append(executor.submit(_process, run, until=stage))
                    results = _join_workers(workers)
                    workers = []
            else:
                if progress:
                    _init_progress(len(self.runs), msg="Processing all stages")
                else:
                    print("Processing all stages")
                for run in self.runs:
                    workers.append(executor.submit(_process, run, until=until))
                results = _join_workers(workers)

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
