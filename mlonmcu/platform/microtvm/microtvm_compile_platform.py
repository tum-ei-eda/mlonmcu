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
"""MicroTVM Compile Platform"""
from pathlib import Path
from typing import Tuple

from mlonmcu.artifact import Artifact, ArtifactFormat

from ..platform import CompilePlatform


class MicroTvmCompilePlatform(CompilePlatform):
    """MicroTVM compile platform class."""

    def invoke_tvmc_micro_create(self, mlf_path, target=None, list_options=False, force=True, **kwargs):
        all_args = []
        if force:
            all_args.append("--force")
        all_args.append(self.project_dir)
        all_args.append(mlf_path)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("create", *all_args, target=target, list_options=list_options, **kwargs)

    def invoke_tvmc_micro_build(self, target=None, list_options=False, force=False, **kwargs):
        all_args = []
        if force:
            all_args.append("--force")
        all_args.append(self.project_dir)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("build", *all_args, target=target, list_options=list_options, **kwargs)

    def prepare(self, mlf, target):
        out = self.invoke_tvmc_micro_create(mlf, target=target)
        return out

    def compile(self, target):
        out = ""
        # TODO: build with cmake options
        out += self.invoke_tvmc_micro_build(target=target)
        return out

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        src = Path(src) / "default.tar"  # TODO: lookup for *.tar file
        artifacts = []
        out = self.prepare(src, target)
        out += self.compile(target)
        stdout_artifact = Artifact(
            "microtvm_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {}
