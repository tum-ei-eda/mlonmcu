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
import tempfile
import itertools
import multiprocessing
from datetime import datetime
from enum import Enum
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time

from enum import IntEnum

from mlonmcu.session.run import Run
from mlonmcu.logging import get_logger
from mlonmcu.report import Report
from mlonmcu.config import filter_config, str2bool

from .postprocess.postprocess import SessionPostprocess
from .run import RunStage
from .progress import init_progress, update_progress, close_progress, get_pbar_callback

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


class SessionStatus(Enum):  # TODO: remove?
    """Status type for a session."""

    CREATED = 0
    OPEN = 1
    CLOSED = 2
    ERROR = 3


def get_cpu_count():
    return multiprocessing.cpu_count()


def get_used_stages(runs, until):
    """Determines the stages which are used by at least one run."""
    if not isinstance(runs, list):
        runs = [runs]
    used = []
    for stage_index in list(range(RunStage.LOAD, until + 1)) + [RunStage.POSTPROCESS]:
        stage = RunStage(stage_index)
        if any(run.has_stage(stage) for run in runs):
            used.append(stage)
    return used


class Group:
    """TODO"""

    def __init__(self, jobs, parent=None):
        self.jobs = jobs
        self.parent = parent

    def __len__(self):
        return len(self, self.jobs)

    def merge(self, runs):
        # TODO: check if valid
        # Run
        new_run = deepcopy(runs[self.jobs[0].run])
        new_run.idx = tuple([job.run for job in self.jobs])
        new_run.hidden = True

        # Job
        new_job = deepcopy(self.jobs[0])
        new_job.idx = tuple([job.idx for job in self.jobs])
        new_job._needs = [self.parent] if self.parent is not None else []
        new_job.run = new_run.idx
        new_job.reset()

        return new_job, new_run


class JobError(Exception):
    pass


class JobStatus(IntEnum):

    DONE = 0
    FAILING = 1
    RUNNING = 2
    BLOCKED = 3
    READY = 4


class Job:
    """TODO."""

    def __init__(self, idx, run, stage=None, needs=None):
        self.idx = idx
        self.run = run  # idx only
        self.stage = stage
        self._needs = needs if needs is not None else []
        self.needs = self._needs.copy()
        self.result = None
        self.in_progress = False
        self.exception = None
        self.failed_stage = None

    def __repr__(self):
        tmp = str(self.run) if self.stage is None else f"{self.run}.{self.stage}"
        # return f"Job({self.idx}, {tmp})"
        return f"Job({self.idx}, {tmp}, {self.needs})"

    @property
    def ready(self):
        return not self.result and not self.exception and not self.in_progress and len(self.needs) == 0

    @property
    def status(self):
        if self.result:
            return JobStatus.DONE
        elif self.exception or self.failed_stage:
            return JobStatus.FAILING
        elif self.in_progress:
            return JobStatus.RUNNING
        elif self.ready:
            return JobStatus.READY
        else:
            return JobStatus.BLOCKED

    def reset(self):
        self.result = None
        self.in_progress = False
        self.exception = None
        self.failed_stage = None
        self.needs = self._needs.copy()

    def process(self, run, until, skip, export):
        assert not self.in_progress  # TODO: just reset instead?
        # print("process", self.idx, self.result, self.exception)
        assert not self.result
        assert not self.exception
        assert self.ready
        self.in_progress = True
        if self.stage:
            # print("self.stage", self.stage, type(self.stage))
            # print("until", until, type(until))
            assert self.stage == until
        try:
            input(f"> {self}")
            # print(f"> {self}")
            result = run.process(until=until, skip=skip, export=export)
            # print(f"< {self}")
            if run.failing:
                if self.stage:
                    self.failed_stage = self.stage.name
                else:
                    self.failed_stage = RunStage(run.next_stage).name
                self.fail(JobError(f"Failed at stage {self.failed_stage}"))
            self.done(result)
        except Exception as e:
            logger.exception(e)
            logger.error("An exception was thrown by a worker during simulation")
            self.fail(e)
        # self.in_progress = False  # See done
        return self

    def update(self, done_list, failed_list):
        # print("update", self.idx, done_list, failed_list, self.needs)
        new = []
        for need in self.needs:
            if need not in done_list:
                new.append(need)
                break
            elif need in failed_list:
                self.fail(JobError("Failed due to failing dependencies"))
                break
        self.needs = new

    def fail(self, exception):
        self.exception = exception
        if not self.failed_stage:
            self.failed_stage = self.stage if self.stage is not None else "UNKNOWN"
        self.in_progress = False

    def done(self, result):
        self.result = result
        self.in_progress = False


def collect_dependencies(jobs):
    deps = []
    for job in jobs:
        for need in job.needs:
            deps.append((need, job.idx))
    return deps


def enumerate_jobs(jobs):
    for i, job in enumerate(jobs):
        job.idx = i


def split_run_to_jobs(run: Run, start=0) -> list:
    jobs = []
    i = start
    prev = None
    for stage in get_used_stages(run, RunStage.DONE):
        needs = [prev] if prev is not None else []
        jobs.append(Job(i, run.idx, stage=stage, needs=needs))
        prev = i
        i = i + 1
    return jobs


def create_groups(jobs, runs, merge=False):
    assert merge
    run_stage_jobs = {}
    for job in jobs:
        run_stage_jobs[(job.run, int(job.stage))] = job.idx
    # print("run_stage_jobs", run_stage_jobs)
    used = {}
    prevs = {}
    prev_stage = {}
    for until in list(range(RunStage.LOAD, RunStage.DONE)):
        mapping = {}
        for run in runs:
            idx = run.idx
            hash_str = str(run.get_hash(until=until))
            if hash_str in mapping:
                mapping[hash_str].append(idx)
            else:
                mapping[hash_str] = [idx]
        # print("mapping", until, mapping)
        # names = ["_".join(list(map(str, value))) for key, value in mapping.items()]
        # keys = tuple(itertools.chain(*mapping.values()))
        mapping = {key: tuple(value) for key, value in mapping.items()}
        for value in mapping.values():
            if value in used:
                used[value].append(until)
            else:
                used[value] = [until]
            # print("name", name)
        # input(">")
    # print("used", used)
    # input(">")
    for key, value in used.items():
        prevs[key] = None
        prev_stage[key] = None
        idxs = key if isinstance(key, tuple) else (key,)
        # print("idxs", idxs)
        for key_, value_ in used.items():
            if key == key_:
                continue
            idxs_ = key_ if isinstance(key_, tuple) else (key_,)
            # print("idxs_", idxs_)
            valid = True
            for idx in idxs:
                if idx not in idxs_:
                    valid = False
                    break
            # print("valid", valid, key, key_)
            if valid:
                prevs[key] = key_
                prev_stage[key] = max(value_)
    # prevs = {key: max(value, key=lambda x:x[1]) if len(value) else [] for key, value in prevs.items()}
    # print("prevs", prevs)
    # print("prev_stages", prev_stage)
    groups_per_stage = {stage: [] for stage in range(RunStage.LOAD, RunStage.DONE)}
    new_jobs = []
    hidden_runs = []
    for keys, values in used.items():
        for stage in values:
            jobs_ = [run_stage_jobs.get((run, stage), None) for run in keys]
            jobs_ = [job for job in jobs_ if job is not None]
            jobs__ = [jobs[job_] for job_ in jobs_]
            parent = prevs.get(keys, None)
            parent_stage = prev_stage.get(keys, None)
            if parent is not None:
                assert parent_stage is not None
                assert parent_stage < stage
                parent = [run_stage_jobs.get((run, parent_stage), None) for run in parent]
                parent = tuple([x for x in parent if x is not None])
            if len(jobs__) == 1:
                new_job = deepcopy(jobs__[0])
                if parent:
                    new_job._needs=[parent]
                    new_job.reset()
                new_jobs.append(new_job)
            elif len(jobs__) > 1:
                # print("keys", keys)
                # print("stage", stage)
                # print("parent", parent)
                group = Group(jobs__, parent=parent)
                job, run = group.merge(runs)
                # print("job", job)
                groups_per_stage[stage].append(group)
                new_jobs.append(job)
                hidden_runs.append(run)
    # print("new_jobs", new_jobs, len(new_jobs))
    # print("groups_per_stage", groups_per_stage)
    input(">")
    return new_jobs, hidden_runs


class JobGraph:
    """TODO."""

    def __init__(self, jobs):
        self.jobs = jobs
        for job in self.jobs:
            job.reset()
        self.nodes = [job.idx for job in self.jobs]
        self.dependencies = collect_dependencies(jobs)

    def plot(self):
        raise NotImplementedError

    @property
    def completed(self):
        # print("completed", len(self.jobs) == len(self.get_done_jobs() + self.get_failing_jobs()))
        return len(self.jobs) == len(self.get_done_jobs() + self.get_failing_jobs())

    def get_jobs(self, status=None, stage=None):
        # print("get_jobs", status, stage)
        if status is None and stage is None:
            return self.jobs
        ret = []
        for job in self.jobs:
            # print("job", job, job.stage, job.status)
            if status is None:
                if job.stage == stage:
                    ret.append(job)
            elif stage is None:
                if job.status == status:
                    ret.append(job)
            elif job.status == status:
                ret.append(job)
        return ret

    def get_running_jobs(self):
        return self.get_jobs(status=JobStatus.RUNNING)

    def get_ready_jobs(self):
        return self.get_jobs(status=JobStatus.READY)

    def get_failing_jobs(self):
        return self.get_jobs(status=JobStatus.FAILING)

    def get_done_jobs(self):
        return self.get_jobs(status=JobStatus.DONE)

    def get_blocked_jobs(self):
        return self.get_jobs(status=JobStatus.BLOCKED)


class Session:
    """A session which wraps around multiple runs in a context."""

    DEFAULTS = {
        "report_fmt": "csv",
        "runs_per_stage": True,
        "interleave_stages": False,
        "group_runs": False,
        # "print_report": True,
    }

    def __init__(self, label="", idx=None, archived=False, dir=None, config=None):
        self.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.label = (
            label if len(label) > 0 else ("unnamed" + "_" + self.timestamp)
        )  # TODO: decide if named sessions should also get a timestamp?
        self.idx = idx
        self.config = config if config else {}
        self.config = filter_config(self.config, "session", self.DEFAULTS, [], [])
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

    def get_run_from_idx(self, idx):
        for run in self.runs:
            if run.idx == idx:
                return run
        return None

    @property
    def prefix(self):
        """get prefix property."""
        return f"[session-{self.idx}] " if self.idx else ""

    @property
    def report_fmt(self):
        """get report_fmt property."""
        return str(self.config["report_fmt"])

    # @property
    # def print_report(self):
    #     """get print_report property."""
    #     value = self.config["print_report"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def runs_per_stage(self):
        """get runs_per_stage property."""
        value = self.config["runs_per_stage"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def interleave_stages(self):
        """get interleave_stages property."""
        value = self.config["interleave_stages"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def group_runs(self):
        """get group_runs property."""
        value = self.config["group_runs"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
                run_idx += 1
        self.next_run_idx = run_idx

    def init_directories(self):
        for run in self.runs:
            print("run", run, run.hidden)
            if not run.archived:
                run.init_directory()


    def request_run_idx(self):
        """Return next free run index."""
        ret = self.next_run_idx
        self.next_run_idx += 1
        # TODO: find a better approach for this
        return ret

    def process_runs(
        self,
        until=RunStage.DONE,
        print_report=False,
        num_workers=1,
        progress=False,
        export=False,
        context=None,
    ):
        """Process a runs in this session until a given stage."""

       # TODO: Add configurable callbacks for stage/run complete


        used_stages = get_used_stages(self.runs, until)
        skipped_stages = [stage for stage in RunStage if stage not in used_stages]

        self.enumerate_runs()
        # TODO: jobs = split_tasks(...)
        per_stage = self.runs_per_stage
        interleave = self.interleave_stages
        # print("interleave", interleave)
        if per_stage:
            jobs = []
            for run in self.runs:
                new = split_run_to_jobs(run, start=len(jobs))
                jobs.extend(new)
        else:
            jobs = [Job(i, run.idx) for i, run in enumerate(self.runs)]

        if self.group_runs:
            jobs, hidden_runs = create_groups(jobs, self.runs, merge=True)
            # print("groups", groups)
        self.runs.extend(hidden_runs)
        # print("!jobs", jobs)
        graph = JobGraph(jobs)
        self.init_directories()
        self.report = None
        assert num_workers > 0, "num_workers can not be < 1"
        workers = []
        # results = []
        workers = []
        pbar = None  # Outer progress bar
        pbar2 = None  # Inner progress bar
        num_runs = len(self.runs)
        num_failures = 0
        stage_failures = {}
        done_jobs = []
        failed_jobs = []

        def _join_workers(workers):
            """Helper function to collect all worker threads."""
            nonlocal num_failures
            results = []
            for i, w in enumerate(workers):
                try:
                    job = w.result()
                    if job.status == JobStatus.DONE:
                        done_jobs.append(job.idx)
                    elif job.status == JobStatus.FAILING:
                        failed_jobs.append(job.idx)
                        failed_stage = job.failed_stage
                        assert failed_stage is not None
                        if failed_stage in stage_failures:
                            stage_failures[failed_stage].append(job.run)
                        else:
                            stage_failures[failed_stage] = [job.run]
                        num_failures += 1
                except Exception as exe:
                    logger.exception(exe)
                    logger.error("An exception was thrown by a worker during simulation")
            return results

        def _process(job, until, skip, export):
            return job.process(self.get_run_from_idx(job.run), until, skip, export)

        with ThreadPoolExecutor(num_workers) as executor:
            if per_stage:
                if interleave:
                    if progress:
                        pbar = init_progress(len(jobs), msg="Processing jobs")

                    submitted = []
                    while not graph.completed:
                        time.sleep(1)
                        ready_jobs = graph.get_ready_jobs()
                        # print("ready_jobs", ready_jobs)
                        # print("submitted", submitted)
                        for job in ready_jobs:
                            # print("job.idx", job.idx)
                            if job.idx in submitted:
                                continue
                            submitted.append(job.idx)
                            future = executor.submit(_process, job, job.stage, skipped_stages, export)
                            # future.add_done_callback(get_pbar_callback(pbar))
                            workers.append(future)
                        assert len(workers) > 0
                        done_list, x = wait(workers, return_when=FIRST_COMPLETED)
                        # print("done_list", done_list)
                        # print("x", x)
                        _join_workers(done_list)
                        # print("done_jobs", done_jobs)
                        # print("failed_jobs", failed_jobs)
                        for job in jobs:
                            job.update(done_jobs, failed_jobs)
                        # print("workers_before", len(workers))
                        for future in done_list:
                            update_progress(pbar)
                            # print("updated")
                            workers.remove(future)
                        # print("workers_after", len(workers))
                    # TODO: detect deadlock (check for circular deps etc.)

                    if progress:
                        close_progress(pbar)
                else:
                    if progress:
                        pbar2 = init_progress(len(used_stages), msg="Processing stages")
                    for stage in used_stages:
                        run_stage = RunStage(stage).name
                        # jobs_ = jobs_per_stage[stage]
                        jobs_ = graph.get_jobs(stage=stage)
                        # print("stage", stage, "jobs_", jobs_, "len(jobs_)", len(jobs))
                        if progress:
                            pbar = init_progress(len(jobs_), msg=f"Processing stage {run_stage}")
                        else:
                            logger.info("%s Processing stage %s", self.prefix, run_stage)
                        for i, job in enumerate(jobs_):
                            if job.failed_stage:
                                logger.warning("Skiping stage '%s' for failed run", run_stage)
                            else:
                                # future = executor.submit(job.process, self.runs[job.run], until=stage, skip=skipped_stages)
                                future = executor.submit(_process, job, stage, skipped_stages, export)
                                # future.add_done_callback(job.callback)
                                future.add_done_callback(get_pbar_callback(pbar))

                                workers.append(future)
                        _join_workers(workers)
                        if progress:
                            close_progress(pbar)
                        # TODO: function
                        for job in jobs:
                            job.update(done_jobs, failed_jobs)
                        workers = []
                        if progress:
                            update_progress(pbar2)
                    if progress:
                        close_progress(pbar2)
            else:
                assert not interleave, "session.interleave_stages requires session.runs_per_stage"
                if progress:
                    pbar = init_progress(len(self.runs), msg="Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for i, job in enumerate(jobs):
                    # workers.append(executor.submit(job.process, self.runs[job.run], until=until, skip=skipped_stages, export=export))
                    future = executor.submit(_process, job, until, skipped_stages, export)
                    future.add_done_callback(get_pbar_callback(pbar))
                    workers.append(future)
                _join_workers(workers)
                if progress:
                    close_progress(pbar)
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
