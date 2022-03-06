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
import tempfile
import os
import psutil
import time
import signal
import random
import multiprocessing
from pathlib import Path

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.setup import utils


def spawn_server(args, env, verbose):
    utils.python("-m", "tvm.exec.rpc_server", *args, live=verbose, env=env)


class TVMTuner:

    DEFAULTS = {
        "enable": False,
        "print_outputs": False,
        "results_file": None,
        "append": None,
        "tuner": "ga",  # Options: ga,gridsearch,random,xgb,xgb_knob,xgb-rank
        "trials": 10,  # TODO: increase to 100?
        "early_stopping": None,  # calculate default dynamically
        "num_workers": 1,
        "max_parallel": 1,
        "use_rpc": False,
        "timeout": 100,
    }

    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config if config is not None else {}
        self.artifact = None
        self.hostname = "127.0.0.1"  # localhost
        self.port = random.randint(9000, 9500)
        self.port_end = self.port + 100
        self.key = "mlonmcu"  # constant
        self.tracker = None
        self.servers = []
        self.pool = None
        # TODO: Support non-local runners in the future -> deploy simulator on cluster

    @property
    def enable(self):
        return bool(self.config["enable"])

    @property
    def results_file(self):
        return self.config["results_file"]

    @property
    def append(self):
        return bool(self.config["append"])

    @property
    def tuner(self):
        tuner = str(self.config["tuner"])
        assert tuner in ["ga", "gridsearch", "random,xgb", "xgb_knob", "xgb-rank"]
        return tuner

    @property
    def trials(self):
        return int(self.config["trials"])

    @property
    def early_stopping(self):
        ret = self.config["early_stopping"]
        if ret is None:
            ret = max(self.trials, 10)  # Let's see if this default works out...
        return int(ret)

    @property
    def num_workers(self):
        return int(self.config["num_workers"])

    @property
    def max_parallel(self):
        return int(self.config["max_parallel"])

    @property
    def use_rpc(self):
        return bool(self.config["use_rpc"])

    @property
    def timeout(self):
        return int(self.config["timeout"])

    @property
    def print_outputs(self):
        return bool(self.config["print_outputs"])

    def get_rpc_args(self):
        return (
            [
                "--rpc-key",
                self.key,
                "--rpc-tracker",
                self.hostname + ":" + str(self.port),
            ]
            if self.use_rpc
            else []
        )

    def get_tvmc_tune_args(self, target="c"):
        args = self.backend.get_common_tvmc_args(target=target)
        args.extend(
            [
                "--tuner",
                self.tuner,
                *(["--early-stopping", str(self.early_stopping)] if self.early_stopping > 0 else []),
                "--parallel",
                str(self.max_parallel),
                "--timeout",
                str(self.timeout * self.max_parallel),
                "--trials",
                str(self.trials),
                *(["--tuning-records", self.results_file] if self.results_file is not None else []),
                *self.get_rpc_args(),
            ]
        )
        return args

    def invoke_tvmc_tune(self, out, verbose=False):
        args = self.get_tvmc_tune_args()
        args.extend(["--output", str(out)])
        self.backend.invoke_tvmc("tune", *args, verbose=verbose)

    def pick_best(self, records, verbose=False):
        content_best = ""
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
            env = self.backend.prepare_python_environment()
            utils.python("-m", "tvm.autotvm.record", *args, live=verbose, env=env)
            with open(out_file, "r") as handle:
                content_best = handle.read()
        return content_best

    def start_rpc_tracker(self, verbose=False):
        assert self.tracker is None
        args = [
            "--host",
            self.hostname,
            "--port",
            str(self.port),
            "--port-end",
            str(self.port_end),
            # --silent not required as muted anyway...
        ]
        env = self.backend.prepare_python_environment()

        def spawn_tracker():
            utils.python("-m", "tvm.exec.rpc_tracker", *args, live=verbose, env=env)

        self.tracker = multiprocessing.Process(target=spawn_tracker)
        self.tracker.start()
        time.sleep(5)

    def shutdown_rpc_tracker(self):
        assert self.tracker is not None and self.tracker.is_alive()
        # self.tracker.terminate()
        pid = self.tracker.pid
        os.kill(pid, signal.SIGINT)

    def start_rpc_servers(self, verbose=False):
        assert len(self.servers) == 0

        args = [
            "--host",
            self.hostname,
            "--port",
            str(self.port),
            "--port-end",
            str(self.port_end),
            "--tracker",
            self.hostname + ":" + str(self.port),
            "--key",
            self.key,
            "--no-fork",  # required?
            # --silent not required as muted anyway...
        ]
        env = self.backend.prepare_python_environment()

        for _ in range(self.max_parallel):
            server = multiprocessing.Process(target=spawn_server, args=(args, env, verbose))
            self.servers.append(server)

        for server in self.servers:
            server.start()

    def shutdown_rpc_servers(self):
        assert len(self.servers) > 0
        for server in self.servers:
            # server.terminate()
            pid = server.pid
            os.kill(pid, signal.SIGINT)

    def setup_rpc(self, verbose=False):
        self.start_rpc_tracker(verbose=verbose)
        self.start_rpc_servers(verbose=verbose)

    def shutdown_rpc(self):
        self.shutdown_rpc_servers()
        self.shutdown_rpc_tracker()

    def tune(self, verbose=False):
        artifacts = []
        if self.print_outputs:
            verbose = True

        content = ""
        if self.enable:
            if self.num_workers > 1:
                raise NotImplementedError("Tuning multiple tasks at once is currently not supported in TVM!")

            # try:
            # TODO: try block with proper cleanup
            if self.use_rpc:
                self.setup_rpc(verbose=verbose)

            if self.append:
                if self.results_file is not None:
                    with open(self.results_file, "r") as handle:
                        content = handle.read()

            with tempfile.TemporaryDirectory() as tmp_dir:
                out_file = Path(tmp_dir) / "tuning_results.log.txt"
                with open(out_file, "w") as handle:
                    handle.write(content)
                    # TODO: newline or not?
                self.invoke_tvmc_tune(out_file, verbose=verbose)
                with open(out_file, "r") as handle:
                    content = handle.read()
            if self.use_rpc:
                self.shutdown_rpc()
        else:
            if self.results_file is None:
                self.artifact = None
                return
            assert Path(self.results_file).is_file()
            with open(self.results_file, "r") as handle:
                content = handle.read()

        artifact = Artifact("tuning_results.log.txt", content=content, fmt=ArtifactFormat.TEXT)
        artifacts.append(artifact)

        # pick best records
        content_best = self.pick_best(content, verbose=verbose)
        if len(content_best) > 0:
            artifact_ = Artifact("best_tuning_results.log.txt", content=content_best, fmt=ArtifactFormat.TEXT)
            artifacts.append(artifact_)

        self.artifacts = artifacts

    def get_results(self):
        # assert self.artifact is not None
        return self.artifacts
