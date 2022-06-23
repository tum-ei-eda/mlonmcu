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
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat


# Warning: This is only ment to be used with the TvmPlatform!


class TVMLLVMBackend(TVMBackend):

    FEATURES = [
        *TVMBackend.FEATURES,
    ]

    DEFAULTS = {
        **TVMBackend.DEFAULTS,
    }

    name = "tvmllvm"

    def get_tvmc_compile_args(self, out):
        return super().get_tvmc_compile_args(out, executor="graph", target="llvm", runtime="cpp", fmt="so")

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
                        fmt=ArtifactFormat.SHARED_OBJECT,
                        archive=True,
                    )
                )

            stdout_artifact = Artifact(
                "tvmc_compile_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmllvm_out.log?
            artifacts.append(stdout_artifact)
        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TVMLLVMBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
