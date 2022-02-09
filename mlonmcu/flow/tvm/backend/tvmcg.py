import sys
import io
import tempfile
import json
from pathlib import Path
import tarfile

from .backend import TVMBackend
from .tvmrt import TVMRTBackend
from .wrapper import generate_wrapper_header
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.setup import utils


class TVMCGBackend(TVMRTBackend):

    name = "tvmcg"

    REQUIRED = TVMRTBackend.REQUIRED + ["utvmcg.exe"]

    def get_max_workspace_size_from_metadata(self, metadata):
        max_workspace = 0
        for op in metadata["memory"]["functions"]["operator_functions"]:
            max_workspace = max(
                max_workspace,
                op["workspace"][0]["workspace_size_bytes"] if len(op["workspace"]) > 0 else 0,
            )
        return max_workspace

    def generate_code(self, verbose=False):
        super().generate_code()
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
                    graph_json_file = mlf_path / "executor-config" / "graph" / "graph.json"
                    params_bin_file = mlf_path / "parameters" / "default.params"
                    max_workspace_size = self.get_max_workspace_size_from_metadata(metadata)
                    args = []
                    args.append(graph_json_file)
                    args.append(params_bin_file)
                    args.append(out_file)
                    args.append(str(max_workspace_size))
                    utils.exec_getout(tvmcg_exe, *args, live=verbose, print_output=False)
                    codegen_src = open(out_file, "r").read()
                    artifact = Artifact("staticrt.c", content=codegen_src, fmt=ArtifactFormat.SOURCE)
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
