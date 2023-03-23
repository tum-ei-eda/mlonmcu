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

from .backend import TVMBackend
from mlonmcu.flow.backend import main
from .tvmc_utils import get_tvmrt_tvmc_args
from .model_info import get_relay_model_info


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

    def get_tvmc_compile_args(self, out, dump=None):
        return super().get_tvmc_compile_args(out, dump=dump) + get_tvmrt_tvmc_args(self.runtime)

    def get_graph_and_params_from_mlf(self, path):
        graph = None
        # with open(Path(path) / "executor-config" / "graph" / "graph.json", "r") as handle:
        with open(Path(path) / "executor-config" / "graph" / "default.graph", "r") as handle:
            graph = handle.read()
        params = None
        with open(Path(path) / "parameters" / "default.params", "rb") as handle:
            params = handle.read()

        return graph, params


if __name__ == "__main__":
    sys.exit(
        main(
            TVMLLVMBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
