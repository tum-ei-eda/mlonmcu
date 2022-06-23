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
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat
from .wrapper import generate_tvmaot_wrapper, generate_wrapper_header
from .tvmc_utils import get_tvmaot_tvmc_args


class TVMAOTBackend(TVMBackend):

    FEATURES = [
        *TVMBackend.FEATURES,
        "debug_arena",
        "unpacked_api",
        "usmp",
    ]

    DEFAULTS = {
        **TVMBackend.DEFAULTS,
        "debug_arena": False,
        "arena_size": None,  # Determined automatically
        "unpacked_api": False,
        "alignment_bytes": 4,
    }

    name = "tvmaot"

    @property
    def arena_size(self):
        size = self.config["arena_size"]
        return int(size) if size is not None else None

    @property
    def unpacked_api(self):
        return bool(self.config["unpacked_api"])

    @property
    def debug_arena(self):
        return bool(self.config["debug_arena"])

    @property
    def alignment_bytes(self):
        return int(self.config["alignment_bytes"])

    def get_tvmc_compile_args(self, out):
        return super().get_tvmc_compile_args(out, executor="aot") + get_tvmaot_tvmc_args(
            self.alignment_bytes, self.unpacked_api
        )

    def get_workspace_size_from_metadata(self, metadata):
        return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]

    def generate_code(self):
        artifacts = []
        assert self.model is not None
        full = False  # Required due to bug in TVM
        dump = ["c", "relay"] if full else []
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / f"{self.prefix}.tar"
            out = self.invoke_tvmc_compile(out_path, dump=dump)
            mlf_path = Path(temp_dir) / "mlf"
            tarfile.open(out_path).extractall(mlf_path)
            with open(mlf_path / "metadata.json") as handle:
                metadata = json.load(handle)
            # metadata_txt = json.dumps(metadata)
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
            if full:  # FIXME: broken due to error in TVM
                with open(str(out_path) + ".c", "r") as handle:
                    mod_src = handle.read()
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.c",
                            content=mod_src,
                            fmt=ArtifactFormat.SOURCE,
                            optional=True,
                        )
                    )
                with open(str(out_path) + ".relay", "r") as handle:
                    mod_txt = handle.read()
                    artifacts.append(
                        Artifact(
                            f"{self.prefix}.relay",
                            content=mod_txt,
                            fmt=ArtifactFormat.TEXT,
                            optional=True,
                        )
                    )
            generate_wrapper = True
            if generate_wrapper:
                if self.arena_size is not None:
                    assert self.arena_size >= 0
                    workspace_size = self.arena_size
                else:
                    workspace_size = self.get_workspace_size_from_metadata(metadata)
                wrapper_src = generate_tvmaot_wrapper(
                    self.model_info,
                    workspace_size,
                    self.prefix,
                    api="c" if self.unpacked_api else "packed",
                    debug_arena=self.debug_arena,
                )
                artifacts.append(Artifact("aot_wrapper.c", content=wrapper_src, fmt=ArtifactFormat.SOURCE))
                header_src = generate_wrapper_header()
                artifacts.append(Artifact("tvm_wrapper.h", content=header_src, fmt=ArtifactFormat.SOURCE))
            workspace_size_artifact = Artifact(
                "tvmaot_workspace_size.txt", content=f"{workspace_size}", fmt=ArtifactFormat.TEXT
            )
            artifacts.append(workspace_size_artifact)
            stdout_artifact = Artifact(
                "tvmc_compile_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmaot_out.log?
            artifacts.append(stdout_artifact)
        # assert self.target
        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TVMAOTBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
