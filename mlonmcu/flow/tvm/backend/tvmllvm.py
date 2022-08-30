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
import sys
import tempfile
import tarfile
from pathlib import Path

from .backend import TVMBackend
from .wrapper import generate_tvmrt_wrapper, generate_wrapper_header
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat
from .tvmc_utils import get_tvmrt_tvmc_args


# Warning: This is only ment to be used with the TvmPlatform!


class TVMLLVMBackend(TVMBackend):
    FEATURES = [
        *TVMBackend.FEATURES,
    ]

    DEFAULTS = {
        **TVMBackend.DEFAULTS,
    }

    name = "tvmllvm"

    def __init__(self, runtime="crt", fmt="mlf", features=None, config=None):
        super().__init__(target="llvm", executor="graph", runtime=runtime, fmt=fmt, features=features, config=config)

    def get_tvmc_compile_args(self, out):
        return super().get_tvmc_compile_args(out) + get_tvmrt_tvmc_args(self.runtime)

    def get_graph_and_params_from_mlf(self, path):
        graph = None
        # with open(Path(path) / "executor-config" / "graph" / "graph.json", "r") as handle:
        with open(Path(path) / "executor-config" / "graph" / "default.graph", "r") as handle:
            graph = handle.read()
        params = None
        with open(Path(path) / "parameters" / "default.params", "rb") as handle:
            params = handle.read()

        return graph, params

    def generate_code(self, verbose=False):
        artifacts = []
        assert self.model is not None

        dump = []
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / f"{self.prefix}.tar"
            out = self.invoke_tvmc_compile(out_path, dump=dump)
            tar_dir = Path(temp_dir) / self.prefix
            tarfile.open(out_path).extractall(tar_dir)

            with open(out_path, "rb") as handle:
                mlf_data = handle.read()
                artifacts.append(
                    Artifact(
                        f"{self.prefix}.tar",
                        raw=mlf_data,
                        fmt=ArtifactFormat.SHARED_OBJECT if self.fmt == "so" else ArtifactFormat.MLF,
                        archive=True,
                    )
                )

            stdout_artifact = Artifact(
                "tvmc_compile_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmllvm_out.log?
            generate_wrapper = self.fmt == "mlf"
            if generate_wrapper:
                workspace_size = 2**20
                assert workspace_size >= 0
                graph, params = self.get_graph_and_params_from_mlf(tar_dir)
                wrapper_src = generate_tvmrt_wrapper(graph, params, self.model_info, workspace_size, debug_arena=False)
                artifacts.append(Artifact("rt_wrapper.c", content=wrapper_src, fmt=ArtifactFormat.SOURCE))
                header_src = generate_wrapper_header()
                artifacts.append(Artifact("tvm_wrapper.h", content=header_src, fmt=ArtifactFormat.SOURCE))
            artifacts.append(stdout_artifact)
        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TVMLLVMBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
