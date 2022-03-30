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
"""Unit tests for the artifact submodule."""

from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts


def test_lookup_artifacts():
    first = Artifact("foo.json", content="{}", fmt=ArtifactFormat.SOURCE, flags={"metadata"})
    second = Artifact("bar.txt", content="bar", fmt=ArtifactFormat.TEXT)
    third = Artifact("hello.elf", raw=b"", fmt=ArtifactFormat.RAW, flags={"sw", "elf", "test"})
    fourth = Artifact("world.txt", content="", fmt=ArtifactFormat.TEXT, flags={"test"})
    artifacts = [first, second, third, fourth]
    assert lookup_artifacts(artifacts) == artifacts
    assert lookup_artifacts(artifacts, first_only=True) == [first]
    assert lookup_artifacts(artifacts, name="foo.txt") == []
    assert lookup_artifacts(artifacts, name="bar.txt") == [second]
    assert lookup_artifacts(artifacts, fmt=ArtifactFormat.TEXT) == [second, fourth]
    assert lookup_artifacts(artifacts, fmt=ArtifactFormat.TEXT, first_only=True) == [second]
    assert lookup_artifacts(artifacts, fmt=ArtifactFormat.RAW) == [third]
    assert lookup_artifacts(artifacts, flags={"test"}) == [third, fourth]
    assert lookup_artifacts(artifacts, flags={"test", "sw"}) == [third]
