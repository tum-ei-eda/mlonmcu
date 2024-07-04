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
from pathlib import Path
from typing import Tuple

from .backend import TVMBackend
from mlonmcu.config import str2bool
from mlonmcu.flow.backend import main
from .wrapper import generate_tvmrt_wrapper, generate_wrapper_header
from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts
from .tvmc_utils import get_tvmrt_tvmc_args
from .model_info import get_relay_model_info


# Warning: This is only ment to be used with the TvmPlatform!


class TVMLLVMBackend(TVMBackend):
    FEATURES = {
        *TVMBackend.FEATURES,
        "debug_arena",
    }

    DEFAULTS = {
        **TVMBackend.DEFAULTS,
        "arena_size": 2**20,  # Can not be detemined automatically (Very large)
        "debug_arena": False,
    }

    name = "tvmllvm"

    def __init__(self, runtime="crt", fmt="mlf", system_lib=True, features=None, config=None):
        super().__init__(
            target="llvm",
            executor="graph",
            runtime=runtime,
            fmt=fmt,
            system_lib=system_lib,
            features=features,
            config=config,
        )

    @property
    def arena_size(self):
        size = self.config["arena_size"]
        return int(size) if size else None

    @property
    def debug_arena(self):
        value = self.config["debug_arena"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def get_tvmc_compile_args(self, out, dump=None):
        return super().get_tvmc_compile_args(out, dump=dump) + get_tvmrt_tvmc_args(
            self.runtime, system_lib=self.system_lib
        )

    def get_graph_and_params_from_mlf(self, path):
        graph = None
        # with open(Path(path) / "executor-config" / "graph" / "graph.json", "r") as handle:
        with open(Path(path) / "executor-config" / "graph" / "default.graph", "r") as handle:
            graph = handle.read()
        params = None
        with open(Path(path) / "parameters" / "default.params", "rb") as handle:
            params = handle.read()

        return graph, params

    def generate(self) -> Tuple[dict, dict]:
        artifacts, metrics = super().generate()
        assert len(artifacts) == 1 and "default" in artifacts
        artifacts = artifacts["default"]
        assert len(metrics) == 1 and "default" in metrics
        metrics = metrics["default"]
        if self.generate_wrapper:
            workspace_size = self.arena_size
            assert workspace_size >= 0
            graph_artifact = lookup_artifacts(artifacts, f"{self.prefix}.graph")[0]
            graph = graph_artifact.content
            params_artifact = lookup_artifacts(artifacts, f"{self.prefix}.params")[0]
            params = params_artifact.raw
            if (not self.model_info) or self.refresh_model_info:
                try:
                    relay_artifact = lookup_artifacts(artifacts, f"{self.prefix}.relay")[0]
                    self.model_info = get_relay_model_info(relay_artifact.content)
                except Exception:
                    assert self.model_info is not None, "Model info missing!"
            wrapper_src = generate_tvmrt_wrapper(
                graph, params, self.model_info, workspace_size, debug_arena=self.debug_arena
            )
            artifacts.append(Artifact("rt_wrapper.c", content=wrapper_src, fmt=ArtifactFormat.SOURCE))
            header_src = generate_wrapper_header()
            artifacts.append(Artifact("tvm_wrapper.h", content=header_src, fmt=ArtifactFormat.SOURCE))
            metrics.add("Workspace Size [B]", workspace_size, True)
        return {"default": artifacts}, {"default": metrics}


if __name__ == "__main__":
    sys.exit(
        main(
            TVMLLVMBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
