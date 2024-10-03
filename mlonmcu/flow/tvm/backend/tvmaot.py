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
import json
from typing import Tuple

from .backend import TVMBackend
from mlonmcu.flow.backend import main
from mlonmcu.config import str2bool
from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts
from .wrapper import generate_tvmaot_wrapper, generate_wrapper_header
from .model_info import get_relay_model_info
from .tvmc_utils import get_tvmaot_tvmc_args


class TVMAOTBackend(TVMBackend):
    FEATURES = {
        *TVMBackend.FEATURES,
        "debug_arena",
        "unpacked_api",
        "usmp",
    }

    DEFAULTS = {
        **TVMBackend.DEFAULTS,
        "debug_arena": False,
        "arena_size": None,  # Determined automatically
        "unpacked_api": False,
        "alignment_bytes": 16,
    }

    name = "tvmaot"

    def __init__(self, runtime="crt", fmt="mlf", system_lib=False, features=None, config=None):
        super().__init__(
            executor="aot", runtime=runtime, fmt=fmt, system_lib=system_lib, features=features, config=config
        )

    @property
    def extra_pass_config(self):
        default = {"tir.usmp.enable": False}
        extra = super().extra_pass_config
        default.update(extra)
        return default

    @property
    def arena_size(self):
        size = self.config["arena_size"]
        return int(size) if size is not None else None

    @property
    def unpacked_api(self):
        value = self.config["unpacked_api"]
        return str2bool(value)

    @property
    def debug_arena(self):
        value = self.config["debug_arena"]
        return str2bool(value)

    @property
    def alignment_bytes(self):
        return int(self.config["alignment_bytes"])

    def get_tvmc_compile_args(self, out, dump=None):
        return super().get_tvmc_compile_args(out, dump=dump) + get_tvmaot_tvmc_args(
            self.alignment_bytes,
            self.unpacked_api,
            system_lib=self.system_lib,
        )

    def get_workspace_size_from_metadata(self, metadata):
        if "modules" in metadata:
            return metadata["modules"]["default"]["memory"]["functions"]["main"][0]["workspace_size_bytes"]
        else:
            # backwards compatibility
            return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]

    def generate(self) -> Tuple[dict, dict]:
        artifacts, metrics = super().generate()
        assert len(artifacts) == 1 and "default" in artifacts
        artifacts = artifacts["default"]
        assert len(metrics) == 1 and "default" in metrics
        metrics = metrics["default"]
        if self.generate_wrapper:
            if self.arena_size is not None:
                assert self.arena_size >= 0
                workspace_size = self.arena_size
            else:
                metadata_artifact = lookup_artifacts(artifacts, f"{self.prefix}.json")[0]
                metadata = json.loads(metadata_artifact.content)
                workspace_size = self.get_workspace_size_from_metadata(metadata)
            if (not self.model_info) or self.refresh_model_info:
                relay_artifact = lookup_artifacts(artifacts, f"{self.prefix}.relay")[0]
                try:
                    self.model_info = get_relay_model_info(relay_artifact.content)
                except Exception:
                    assert self.model_info is not None, "Model info missing!"
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
        metrics.add("Workspace Size [B]", workspace_size, True)
        return {"default": artifacts}, {"default": metrics}


if __name__ == "__main__":
    sys.exit(
        main(
            TVMAOTBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
