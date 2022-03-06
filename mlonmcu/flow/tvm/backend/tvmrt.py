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
import json
import tarfile
from pathlib import Path

from .backend import TVMBackend
from .wrapper import generate_tvmrt_wrapper, generate_wrapper_header
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat


class TVMRTBackend(TVMBackend):

    FEATURES = [
        *TVMBackend.FEATURES,
        "debug_arena",
    ]

    DEFAULTS = {
        **TVMBackend.DEFAULTS,
        "arena_size": 2 ** 20,  # Can not be detemined automatically (Very large)
        # TODO: arena size warning!
    }

    name = "tvmrt"

    @property
    def arena_size(self):
        size = self.config["arena_size"]
        return int(size) if size else None

    def get_tvmc_compile_args(self):
        return super().get_tvmc_compile_args("graph") + [
            "--runtime-crt-system-lib",
            str(1),
            "--executor-graph-link-params",
            str(0),
        ]

    def get_graph_and_params_from_mlf(self, path):
        graph = None
        with open(Path(path) / "executor-config" / "graph" / "graph.json", "r") as handle:
            graph = handle.read()
        params = None
        with open(Path(path) / "parameters" / "default.params", "rb") as handle:
            params = handle.read()

        return graph, params

    def generate_code(self, verbose=False):
        artifacts = []
        assert self.model is not None
        full = False  # Required due to bug in TVM
        dump = ["c", "relay"] if full else []
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / f"{self.prefix}.tar"
            self.invoke_tvmc_compile(out_path, dump=dump, verbose=verbose)
            mlf_path = Path(temp_dir) / "mlf"
            tarfile.open(out_path).extractall(mlf_path)
            with open(mlf_path / "metadata.json") as handle:
                metadata = json.load(handle)
            metadata_txt = json.dumps(metadata)
            with open(out_path, "rb") as handle:
                mlf_data = handle.read()
                artifacts.append(
                    Artifact(
                        f"{self.prefix}.tar",
                        raw=mlf_data,
                        fmt=ArtifactFormat.MLF,
                        archive=True,
                    )
                )
            if full:
                with open(str(out_path) + ".c", "r") as handle:
                    mod_src = handle.read()
                    artifacts.append(
                        Artifac(
                            f"{self.prefix}.c",
                            content=mod_str,
                            fmt=ArtifactFormat.SOURCE,
                            optional=True,
                        )
                    )
                with open(str(out_path) + ".relay", "r") as handle:
                    mod_txt = handle.read()
                    artifacts.append(
                        Artifac(
                            f"{self.prefix}.relay",
                            content=mod_txt,
                            fmt=ArtifactFormat.TEXT,
                            optional=True,
                        )
                    )
            generate_wrapper = True
            if generate_wrapper:
                workspace_size = self.arena_size
                assert workspace_size >= 0
                graph, params = self.get_graph_and_params_from_mlf(mlf_path)
                wrapper_src = generate_tvmrt_wrapper(graph, params, self.model_info, workspace_size)
                artifacts.append(Artifact("rt_wrapper.c", content=wrapper_src, fmt=ArtifactFormat.SOURCE))
                header_src = generate_wrapper_header()
                artifacts.append(Artifact("tvm_wrapper.h", content=header_src, fmt=ArtifactFormat.SOURCE))

        # prepare -> common?
        # invoke_tvmc -> common?
        # generate_wrapper()
        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TVMRTBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
