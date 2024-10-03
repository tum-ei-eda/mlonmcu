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
import os
import tempfile
from pathlib import Path
from typing import Tuple

from .backend import TFLMBackend
import mlonmcu.setup.utils as utils
from mlonmcu.config import str2bool
from mlonmcu.flow.backend import main
from mlonmcu.logging import get_logger
from mlonmcu.artifact import Artifact, ArtifactFormat

logger = get_logger()


class TFLMCBackend(TFLMBackend):
    name = "tflmc"

    FEATURES = {"debug_arena"}

    DEFAULTS = {
        **TFLMBackend.DEFAULTS,
        "print_outputs": False,
        "custom_ops": [],
        "registrations": {},
        "debug_arena": False,
    }

    REQUIRED = TFLMBackend.REQUIRED | {"tflmc.exe"}

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)
        self.model_data = None
        self.prefix = "model"  # Without the _
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data

    def generate_header(self):
        upper_prefix = self.prefix.upper()
        code = f"""
// This file is generated. Do not edit.
#ifndef {upper_prefix}_GEN_H
#define {upper_prefix}_GEN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

void model_init();
void *model_input_ptr(int index);
size_t model_input_size(int index);
size_t model_inputs();
void model_invoke();
void *model_output_ptr(int index);
size_t model_output_size(int index);
size_t model_outputs();

#ifdef __cplusplus
}}
#endif

#endif  // {upper_prefix}_GEN_H
"""
        return code

    def generate(self) -> Tuple[dict, dict]:
        artifacts = []
        assert self.model is not None
        tflmc_exe = None
        if "tflmc.exe" in self.config:
            tflmc_exe = self.config["tflmc.exe"]
        else:
            # Lookup cache
            raise NotImplementedError
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.debug("Using temporary directory for codegen results: %s", tmpdirname)
            args = []
            args.append(str(self.model))
            args.append(str(Path(tmpdirname) / f"{self.prefix}.cc"))
            args.append(f"{self.prefix}_")
            out = utils.execute(tflmc_exe, *args, live=self.print_outputs)
            files = [f for f in os.listdir(tmpdirname) if os.path.isfile(os.path.join(tmpdirname, f))]
            # TODO: ensure that main file is processed first
            for filename in files:
                with open(Path(tmpdirname) / filename, "r") as handle:
                    content = handle.read()
                    artifacts.append(Artifact(filename, content=content, fmt=ArtifactFormat.SOURCE))
            header_content = self.generate_header()
            header_artifact = Artifact(f"{self.prefix}.cc.h", content=header_content, fmt=ArtifactFormat.SOURCE)
            artifacts.append(header_artifact)
            stdout_artifact = Artifact("tflmc_out.log", content=out, fmt=ArtifactFormat.TEXT)
            artifacts.append(stdout_artifact)

        return {"default": artifacts}, {}


if __name__ == "__main__":
    sys.exit(
        main(
            TFLMCBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
