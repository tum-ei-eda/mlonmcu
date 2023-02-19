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
import tempfile
from pathlib import Path

from .loader import load_environment_from_file
from .writer import write_environment_to_file
from .environment import DefaultEnvironment


class UserEnvironment(DefaultEnvironment):
    def __init__(
        self,
        home,
        alias=None,
        defaults=None,
        paths=None,
        repos=None,
        frameworks=None,
        backends=None,
        frontends=None,
        platforms=None,
        toolchains=None,
        targets=None,
        features=None,
        postprocesses=None,
        variables=None,
        default_flags=None,
    ):
        super().__init__()
        self._home = home

        if alias:
            self.alias = alias
        if defaults:
            self.defaults = defaults
        if paths:
            self.paths = paths
        if repos:
            self.repos = repos
        if frameworks:
            self.frameworks = frameworks
        if backends:
            self.backends = backends
        if frontends:
            self.frontends = frontends
        if platforms:
            self.platforms = platforms
        if toolchains:
            self.toolchains = toolchains
        if targets:
            self.targets = targets
        if features:
            self.features = features
        if postprocesses:
            self.postprocesses = postprocesses
        if variables:
            self.vars = variables
        if default_flags:
            self.flags = default_flags

    @classmethod
    def from_file(cls, filename):
        return load_environment_from_file(filename, base=cls)

    def to_file(self, filename):
        write_environment_to_file(self, filename)

    def to_text(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            dest = Path(tmpdirname) / "out.yml"
            self.to_file(dest)
            with open(dest, "r") as handle:
                content = handle.read()
        assert len(content) > 0
        return content
