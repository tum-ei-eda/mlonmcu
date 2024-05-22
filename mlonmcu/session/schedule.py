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
import random
import concurrent.futures
from pathlib import Path
from typing import List, Optional

# from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.run import Run, RunInitializer, RunResult, RunStage
from mlonmcu.logging import get_logger
from mlonmcu.setup import utils

from .postprocess.postprocess import SessionPostprocess
from .progress import init_progress, update_progress, close_progress
from .rpc import connect_tracker, RemoteConfig

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# class SessionExecutor(ABC):
#     def submit_runs(self, ?):
#         raise NotImplementedError


class ProcessPoolSessionExecutor(concurrent.futures.ProcessPoolExecutor):

    def __init__(self, max_workers: Optional[int] = None):
        super().__init__(max_workers)

    def submit_runs(
        self,
        runs,
        until=None,
        skip=None,
        export=False,
        context=None,
        runs_dir=None,
        save=True,
        cleanup=False,
    ):
        fn = _process_pickable
        args = [
            runs,
        ]
        kwargs = {
            "until": until,
            "skip": skip,
            "export": export,
            "context": context,
            "runs_dir": runs_dir,
            "save": save,
            "cleanup": cleanup,
        }
        return self.submit(fn, *args, **kwargs)


class ThreadPoolSessionExecutor(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, max_workers: Optional[int] = None):
        super().__init__(max_workers)

    def submit_runs(
        self,
        runs,
        until=None,
        skip=None,
        export=False,
        context=None,
        runs_dir=None,
        save=True,
        cleanup=False,
    ):
        fn = _process_default
        args = [
            runs,
        ]
        kwargs = {
            "until": until,
            "skip": skip,
            "export": export,
            "context": context,
            "runs_dir": runs_dir,
            "save": save,
            "cleanup": cleanup,
        }
        return self.submit(fn, *args, **kwargs)


class CmdlineSessionExecutor(concurrent.futures.ProcessPoolExecutor):

    def __init__(self, max_workers: Optional[int] = None, parallel_jobs: int = 1):
        super().__init__(max_workers)
        self.parallel_jobs = parallel_jobs

    # def submit_batch(self, fn, /, *args, **kwargs):
    #    fn =
    # def get_process(self):
        # TODO: move args
    def submit_runs(
        self,
        runs,
        until=None,
        skip=None,
        export=False,
        context=None,
        runs_dir=None,
        save=True,
        cleanup=False,
    ):
        fn = _process_cmdline
        args = [
            runs,
        ]
        kwargs = {
            "until": until,
            "skip": skip,
            "export": export,
            "context": context,
            "runs_dir": runs_dir,
            "save": save,
            "cleanup": cleanup,
            "parallel_jobs": self.parallel_jobs,
        }
        return self.submit(fn, *args, **kwargs)


class ContextSessionExecutor(concurrent.futures.ProcessPoolExecutor):

    def __init__(self, max_workers: Optional[int] = None, parallel_jobs: int = 1):
        super().__init__(max_workers)
        self.parallel_jobs = parallel_jobs

    def submit_runs(
        self,
        runs,
        until=None,
        skip=None,
        export=False,
        context=None,
        runs_dir=None,
        save=True,
        cleanup=False,
    ):
        fn = _process_context
        # TODO: do not realize jobs (just dump initializer)
        args = [
            runs,
        ]
        kwargs = {
            "until": until,
            "skip": skip,
            "export": export,
            # "context": context,
            "runs_dir": runs_dir,
            "save": save,
            "cleanup": cleanup,
            "parallel_jobs": self.parallel_jobs,
        }
        return self.submit(fn, *args, **kwargs)


class RPCSessionExecutor(concurrent.futures.ProcessPoolExecutor):

    def __init__(
        self,
        max_workers: Optional[int] = None,
        remote_config: Optional[RemoteConfig] = None,
        blocking: bool = True,
        # batch_size?
        parallel_jobs: int = 1
    ):
        super().__init__(max_workers)
        self.remote_config = remote_config
        self.blocking = blocking
        self.parallel_jobs = parallel_jobs

    def submit_runs(
        self,
        runs,
        until=None,
        skip=None,
        export=False,
        context=None,
        runs_dir=None,
        save=True,
        cleanup=False,
    ):
        fn = _process_rpc
        args = [
            runs,
        ]
        kwargs = {
            "until": until,
            # "skip": skip,
            # "export": export,
            # "context": context,
            # "runs_dir": runs_dir,
            # "save": save,
            # "cleanup": cleanup,
            "parallel_jobs": self.parallel_jobs,
            "rpc_config": self.remote_config,
        }
        return self.submit(fn, *args, **kwargs)


def _handle_context(context, allow_none: bool = False, minimal: bool = False):
    # TODO: handle (thread_pool, process_pool, remote, hybrid)
    if context is None:
        assert allow_none, "Missing context"
        return None
    if minimal:
        return context.get_read_only_context()
    return context


def _process_default(runs, until, skip, export, context, runs_dir, save, cleanup):
    """Helper function to invoke the run."""
    assert isinstance(runs, list)
    rets = []
    for run in runs:
        run.process(until=until, skip=skip, export=export)
        ret = run.result()
        rets.append(ret)
        if save:
            # run.save(run.dir / "run.pkl")
            # run.save_artifacts(run.dir / "artifacts.pkl")
            run.save_artifacts(run.dir / "artifacts.yml")
        if cleanup:
            run.cleanup_artifacts(dirs=True)
            run.cleanup_directories()
    return rets


def _process_pickable(run_initializers, until, skip, export, context, runs_dir, save, cleanup):
    """Helper function to invoke the run."""
    rets = []
    for run_initializer in run_initializers:
        run = run_initializer.realize(context=context)
        run.init_directory(parent=runs_dir)
        run_initializer.save(run.dir / "initializer.yml")
        used_stages = _used_stages([run], until)
        assert skip is None
        skip = [stage for stage in RunStage if stage not in used_stages]
        run.process(until=until, skip=skip, export=export)
        ret = run.result()
        rets.append(ret)
        save = True
        if save:
            # run.save(run.dir / "run.pkl")
            # run.save_artifacts(run.dir / "artifacts.pkl")
            run.save_artifacts(run.dir / "artifacts.yml")
        cleanup = True
        if cleanup:
            run.cleanup_artifacts(dirs=True)
            run.cleanup_directories()
    return rets


def _process_cmdline(run_initializers, until, skip, export, context, runs_dir, save, cleanup, parallel_jobs):
    """Helper function to invoke the run."""
    rets = []
    # parallel_jobs = 1
    args = ["-m", "mlonmcu.cli.main", "flow", until.name.lower(), "_", "--parallel", str(parallel_jobs), "-c", "runs_per_stage=0", "-c", "session.use_init_stage=1", "--initializer"]
    for run_initializer in run_initializers:
        run = run_initializer.realize(context=context)
        run.init_directory(parent=runs_dir)
        run_initializer.save(run.dir / "initializer.yml")
        args.append(run.dir / "initializer.yml")
        res = None  # TODO
        rets.append(res)
    out = utils.python(*args)
    return rets


def _process_context(run_initializers, until, skip, export, runs_dir, save, cleanup, parallel_jobs):
    """Helper function to invoke the run."""
    rets = []
    # TODO: allow overriding home
    from mlonmcu.context.context import MlonMcuContext
    with MlonMcuContext(deps_lock="read") as context:
        with context.get_session(resume=False) as session:
            for run_initializer in run_initializers:
                session.add_run(run_initializer, ignore_idx=True)
                rets.append(None)
            success = session.process_runs(
                until=until,
                per_stage=False,
                print_report=False,
                num_workers=parallel_jobs,
                progress=False,
                context=context,
                export=True,
            )
        assert success
    return rets


def _process_rpc(run_initializers, until, parallel_jobs, rpc_config):
    """Helper function to invoke the run."""
    # TODO: allow overriding home
    tracker = connect_tracker(rpc_config.tracker_host, rpc_config.tracker_port, check=True)
    server = tracker.request_server(key=rpc_config.key)
    results = server.execute(initializers=run_initializers, until=until, parallel=parallel_jobs)
    # -> msg: {"action": "execute", "initializers": run_initializers, "until": until, "parallel": parallel_jobs}
    # <- msg: {"action": "response", "results": results}
    return results


def _postprocess_default(runs, report, dest, progress=False):
    session_postprocesses = []
    num_failing = 0
    for run in runs:
        for postprocess in run.postprocesses:
            if isinstance(postprocess, SessionPostprocess):
                if postprocess.name not in [p.name for p in session_postprocesses]:
                    session_postprocesses.append(postprocess)
    if progress:
        pbar = init_progress(len(session_postprocesses), msg="Postprocessing session")
    for postprocess in session_postprocesses:
        try:
            artifacts = postprocess.post_session(report)
        except Exception as e:
            logger.exception(e)
            num_failing += 1
            break
        if progress:
            update_progress(pbar)
        if artifacts is not None:
            for artifact in artifacts:
                # Postprocess has an artifact: write to disk!
                logger.debug("Writing postprocess artifact to disk: %s", artifact.name)
                artifact.export(dest)
    if progress:
        close_progress(pbar)
    return num_failing


def _postprocess_pickable(runs, report, dest, progress=False):
    logger.error("Session Postprocesses are not supported in pickable mode!")
    return 0


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
        shuffle: bool = False,
        batch_size: int = 1,
        parallel_jobs: int = 1,
        remote_config: Optional[RemoteConfig] = None,
        use_init_stage: bool = False,
        prefix: Optional[str] = None,
        runs_dir: Optional[Path] = None,
        session=None,  # TODO: typing
    ):
        self.runs = runs
        self.results = [None] * len(runs)
        self.until = until
        self.per_stage = per_stage
        self.progress = progress
        self.executor = executor
        self.num_workers = num_workers
        self.parallel_jobs = parallel_jobs
        self._executor_cls, self._executor_kwargs = self._handle_executor(executor, remote_config)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.prefix = session.prefix if session is not None else prefix
        self.runs_dir = session.runs_dir if session is not None else runs_dir
        self.use_init_stage = use_init_stage
        self._futures = []
        # TODO: contextmanager?
        self.num_failures = 0
        self.stage_failures = {}
        # worker_run_idx = []
        # self._future_run_idx = {}
        self._future_batch_idx = {}
        self._batch_run_idxs = {}
        self._check()
        self.used_stages, self.skipped_stages = self.prepare()
        # self._process, self._postprocess = self._pick_process()
        _, self._postprocess = self._pick_process()

    def _reset_futures(self):
        self._futures = []
        # self._future_run_idx = {}
        self._future_batch_idx = {}
        self._batch_run_idxs = {}

    def _handle_executor(self, name: str, remote_config: Optional[RemoteConfig] = None):
        # TODO: handle (thread_pool, process_pool, remote, hybrid)
        EXECUTOR_LOOKUP = {
            "thread_pool": ThreadPoolSessionExecutor,
            "process_pool": ProcessPoolSessionExecutor,
            "popen_pool": None,  # TODO
            "cmdline": CmdlineSessionExecutor,
            "context": ContextSessionExecutor,
            "rpc": RPCSessionExecutor,
        }
        ret = EXECUTOR_LOOKUP.get(name, None)
        assert ret is not None, f"Executor not found: {name}"
        kwargs = {"max_workers": self.num_workers}
        if name in ["cmdline", "context", "rpc"]:
            kwargs["parallel_jobs"] = self.parallel_jobs
        if name in ["rpc"]:
            kwargs["blocking"] = True
            kwargs["remote_config"] = remote_config
        else:
            assert self.parallel_jobs == 1
        return ret, kwargs

    @property
    def _prefix(self):
        return f"{self.prefix} " if self.prefix else ""

    @property
    def use_batches(self):
        return self.batch_size > 1

    def _pick_process(self):
        ret = None
        ret2 = _postprocess_default
        needs_pickable = self.executor in ["process_pool", "cmdline", "context"]
        if needs_pickable:
            # ret = _process_pickable
            ret2 = _postprocess_pickable
        # elif self.executor == "cmdline":
        #     ret = _process_cmdline
        return ret, ret2

    def _check(self):
        has_initializer = False
        for run in self.runs:
            if isinstance(run, RunInitializer):
                has_initializer = True
                break
        if has_initializer:
            assert self.executor in ["process_pool", "cmdline", "context"] or self.use_init_stage
            # raise RuntimeError("RunInitializer needs init stage or process_pool executor")  # TODO: change default
        if self.executor in ["process_pool", "cmdline", "context"]:
            # assert not self.progress, "progress bar not supported if session.process_pool=1"
            assert not self.per_stage, "per stage not supported if session.process_pool=1"
            assert not self.use_init_stage, "use_init_stage not supported if session.process_pool=1"

    def prepare(self):
        if self.executor in ["process_pool", "cmdline", "context"] or self.use_init_stage:
            return None, None  # TODO
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
        for f in concurrent.futures.as_completed(self._futures):
            res = None
            failing = False
            batch_res = None
            try:
                batch_res = f.result()
                assert isinstance(batch_res, list)
            except Exception as e:
                failing = True
                logger.exception(e)
                logger.error("An exception was thrown by a worker during simulation")
            if self.progress:
                update_progress(pbar)
            batch_index = self._future_batch_idx[f]
            run_idxs = self._batch_run_idxs[batch_index]
            if failing:
                assert batch_res is None
                self.num_failures += len(run_idxs)
                failed_stage = None
                if failed_stage in self.stage_failures:
                    self.stage_failures[failed_stage] += run_idxs
                else:
                    self.stage_failures[failed_stage] = [*run_idxs]
            else:
                assert len(batch_res) == len(run_idxs)
                for res_idx, res in enumerate(batch_res):
                    if res is not None:
                        assert isinstance(res, RunResult), "Expected RunResult type"
                        run_index = res.idx
                        assert run_index == run_idxs[res_idx]
                        # run = res
                        # self.runs[run_index] = res
                        self.results[run_index] = res
                    else:
                        assert False, "Should not be used?"
                    run = self.runs[run_index]
                    if res.failing:
                        self.num_failures += 1
                        failed_stage = RunStage(run.next_stage).name if isinstance(run, Run) else None  # TODO
                        if failed_stage in self.stage_failures:
                            self.stage_failures[failed_stage].append(run_index)
                        else:
                            self.stage_failures[failed_stage] = [run_index]
        self._reset_futures()
        if self.progress:
            close_progress(pbar)

    def initialize(self, context):
        runs = []
        if self.progress:
            pbar = init_progress(self.num_runs, msg="Initializing all runs")
        for run_initializer in self.runs:
            assert isinstance(run_initializer, RunInitializer)
            run = run_initializer.realize(context=context)
            run.init_directory(parent=self.runs_dir)
            run_initializer.save(run.dir / "initializer.yml")
            runs.append(run)
            if self.progress:
                update_progress(pbar)
        self.runs = runs
        if self.used_stages is None:
            assert self.skipped_stages is None
            self.used_stages = _used_stages(self.runs, self.until)
            self.skipped_stages = [stage for stage in RunStage if stage not in self.used_stages]
        if self.progress:
            close_progress(pbar)

    def process(
        self,
        export=False,
        context=None,
    ):
        pbar = None  # Outer progress bar
        pbar2 = None  # Inner progress bar
        context_ = _handle_context(context, minimal=True)

        # TODO: expose
        save = True
        cleanup = False  # incompatible with per_stage

        if self.use_init_stage:
            self.initialize(context)

        run_it = [*self.runs]
        if self.shuffle:
            run_it = sorted(run_it, key=lambda _: random.random())
        batches = list(chunks(run_it, self.batch_size))
        # TODO: per stage batching?
        with self._executor_cls(**self._executor_kwargs) as executor:
            if self.per_stage:
                assert self.used_stages is not None
                if self.progress:
                    pbar2 = init_progress(len(self.used_stages), msg="Processing stages")
                for stage in self.used_stages:
                    run_stage = RunStage(stage).name
                    if self.progress:
                        pbar = init_progress(len(batches), msg=f"Processing stage {run_stage}")
                    else:
                        logger.info("%sProcessing stage %s", self._prefix, run_stage)
                    for b, runs in enumerate(batches):
                        runs_ = [run for run in runs if not run.failing]
                        idxs = [run.idx for run in runs_]
                        assert len(runs) > 0
                        if len(runs_) < len(runs):
                            logger.warning("Skiping stage '%s' for failed run", run_stage)
                        else:
                            f = executor.submit_runs(
                                runs_,
                                until=stage,
                                skip=self.skipped_stages,
                                export=export,
                                context=context_,
                                runs_dir=self.runs_dir,
                                save=save,
                                cleanup=cleanup,
                            )
                            self._futures.append(f)
                            # self._future_run_idx[f] = i
                            self._future_batch_idx[f] = b
                            self._batch_run_idxs[b] = idxs
                    self._join_futures(pbar)
                    if self.progress:
                        update_progress(pbar2)
                if self.progress:
                    close_progress(pbar2)
            else:
                if self.progress:
                    pbar = init_progress(len(batches), msg="Processing batches" if self.use_batches else "Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for b, runs in enumerate(batches):
                    idxs = [run.idx for run in runs]
                    assert len(runs) > 0
                    f = executor.submit_runs(
                        runs,
                        until=self.until,
                        skip=self.skipped_stages,
                        export=export,
                        context=context_,
                        runs_dir=self.runs_dir,
                        save=save,
                        cleanup=cleanup,
                    )
                    self._futures.append(f)
                    # self._future_run_idx[f] = i
                    self._future_batch_idx[f] = b
                    self._batch_run_idxs[b] = idxs
                self._join_futures(pbar)
        return self.runs, self.results
        # return num_failures == 0

    def postprocess(self, report, dest):
        logger.info("Postprocessing session report")
        # Warning: currently we only support one instance of the same type of postprocess,
        # also it will be applied to all rows!
        num_failing = self._postprocess(self.runs, report, dest, progress=self.progress)
        self.num_failures += num_failing
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
