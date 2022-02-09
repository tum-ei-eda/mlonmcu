import tempfile
import multiprocessing
from pathlib import Path

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.setup import utils


class TVMTuner:

    DEFAULTS = {
        "enable": False,
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
        self.port = 9000
        self.port_end = 10000
        self.key = "mlonmcu"  # constant
        self.tracker = None
        self.servers = []
        # TODO: Support non-local runners in the future -> deploy simulator on cluster

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

    def get_rpc_args(self):
        print("get_rpc_args", self.use_rpc)
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

    def invoke_tvmc_tune(self, out, dump=None, verbose=False):
        args = self.get_tvmc_tune_args()
        args.extend(["--output", str(out)])
        self.backend.invoke_tvmc("tune", *args, verbose=verbose)

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

    def shutdown_rpc_tracker(self):
        assert self.tracker is not None and self.tracker.is_alive()
        self.tracker.terminate()

    def start_rpc_servers(self, verbose=False):
        assert len(self.servers) == 0

        def spawn_server():
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
                # --silent not required as muted anyway...
            ]
            env = self.backend.prepare_python_environment()
            utils.python("-m", "tvm.exec.rpc_server", *args, live=verbose, env=env)

        for _ in range(self.max_parallel):
            server = multiprocessing.Process(target=spawn_server)
            self.servers.append(server)

        for server in self.servers:
            server.start()

    def shutdown_rpc_servers(self):
        assert len(self.servers) > 0
        for server in self.servers:
            server.terminate()

    def setup_rpc(self, verbose=False):
        self.start_rpc_tracker(verbose=verbose)
        self.start_rpc_servers(verbose=verbose)

    def shutdown_rpc(self):
        self.shutdown_rpc_servers()
        self.shutdown_rpc_tracker()

    def tune(self):
        if self.num_workers > 1:
            raise NotImplementedError("Tuning multiple tasks at once is currently not supported in TVM!")

        # TODO: try block with proper cleanup
        if self.use_rpc:
            self.setup_rpc()

        content = ""
        if self.append:
            if self.results_file is not None:
                with open(self.results_file, "r") as handle:
                    content = handle.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "tuning_results.log.txt"
            with open(out_file, "w") as handle:
                handle.write(content)
                # TODO: newline or not?
            self.invoke_tvmc_tune(out_file)
            with open(out_file, "r") as handle:
                content = handle.read()

        if self.use_rpc:
            self.shutdown_rpc()

        artifact = Artifact("tuning_results.log.txt", content=content, fmt=ArtifactFormat.TEXT)

        self.artifact = artifact

    def get_results(self):
        # assert self.artifact is not None
        return self.artifact
