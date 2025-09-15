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
import io
import tempfile
import json
from pathlib import Path
import tarfile
from typing import Tuple

from .tvmrt import TVMRTBackend
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.setup import utils


class TVMCGBackend(TVMRTBackend):
    name = "tvmcg"

    REQUIRED = TVMRTBackend.REQUIRED | {"utvmcg.exe"}

    def get_max_workspace_size_from_metadata(self, metadata):
        max_workspace = 0
        if "modules" in metadata:
            temp = metadata["modules"]["default"]["memory"]
        else:
            # backwards compatibility
            temp = metadata["memory"]
        for op in temp["functions"]["operator_functions"]:
            max_workspace = max(
                max_workspace,
                op["workspace"][0]["workspace_size_bytes"] if len(op["workspace"]) > 0 else 0,
            )
        return max_workspace

    def generate(self) -> Tuple[dict, dict]:
        super().generate()
        artifacts = self.artifacts
        artifact = None
        for artifact in artifacts:
            if artifact.fmt == ArtifactFormat.MLF:
                with tempfile.TemporaryDirectory() as temp_dir:
                    out_file = Path(temp_dir) / "staticrt.c"
                    mlf_path = Path(temp_dir) / "mlf"
                    mlf_bytes = io.BytesIO(artifact.raw)
                    tarfile.open(fileobj=mlf_bytes, mode="r").extractall(mlf_path)
                    artifact.export(mlf_path, extract=True)
                    metadata = None
                    with open(mlf_path / "metadata.json") as handle:
                        metadata = json.load(handle)
                    tvmcg_exe = self.config["utvmcg.exe"]
                    # graph_json_file = mlf_path / "executor-config" / "graph" / "graph.json"
                    graph_json_file = mlf_path / "executor-config" / "graph" / "default.graph"
                    params_bin_file = mlf_path / "parameters" / "default.params"
                    max_workspace_size = self.get_max_workspace_size_from_metadata(metadata)
                    args = []
                    args.append(graph_json_file)
                    args.append(params_bin_file)
                    args.append(out_file)
                    args.append(str(max_workspace_size))
                    out = utils.execute(tvmcg_exe, *args, live=self.print_outputs)
                    codegen_src = open(out_file, "r").read()
                    artifact = Artifact("staticrt.c", content=codegen_src, fmt=ArtifactFormat.SOURCE)
                    workspace_size_artifact = Artifact(
                        "tvmcg_workspace_size.txt", content=f"{max_workspace_size}", fmt=ArtifactFormat.TEXT
                    )
                    artifacts.append(workspace_size_artifact)
                    stdout_artifact = Artifact("tvmcg_out.log", content=out, fmt=ArtifactFormat.TEXT)
                    artifacts.append(stdout_artifact)
                break
        assert artifact is not None, "Failed to find MLF artifact"
        artifacts.append(artifact)
        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TVMCGBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
