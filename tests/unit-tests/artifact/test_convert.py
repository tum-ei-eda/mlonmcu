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

import pytest

from mlonmcu.artifact import Artifact, ArtifactFormat


def test_convert_artifacts_unknown_valid():
    # UNKNOWN -> UNKNOWN
    in_artifact = Artifact("foo.bar", fmt=ArtifactFormat.UNKNOWN)
    out_artifact = in_artifact.convert(ArtifactFormat.UNKNOWN)
    out_artifact.validate()
    assert out_artifact.fmt == ArtifactFormat.UNKNOWN
    # TODO: check for equality


def test_convert_artifacts_unknown_invalid():
    # UNKNOWN -> *
    in_artifact = Artifact("foo.bar", fmt=ArtifactFormat.UNKNOWN)
    with pytest.raises(NotImplementedError):
        _ = in_artifact.convert(ArtifactFormat.SOURCE)


# @pytest.mark.parametrize("from_", [ArtifactFormat.SOURCE, ArtifactFormat.TEXT])
# @pytest.mark.parametrize("to", [ArtifactFormat.SOURCE, ArtifactFormat.TEXT])
@pytest.mark.parametrize("from_,to", [
    # same
    (ArtifactFormat.SOURCE, ArtifactFormat.SOURCE),
    (ArtifactFormat.TEXT, ArtifactFormat.TEXT),
    # cross
    (ArtifactFormat.SOURCE, ArtifactFormat.SOURCE),
    (ArtifactFormat.TEXT, ArtifactFormat.TEXT),
    # source/text -> raw/bin
    (ArtifactFormat.SOURCE, ArtifactFormat.RAW),
    (ArtifactFormat.SOURCE, ArtifactFormat.BIN),
    (ArtifactFormat.TEXT, ArtifactFormat.RAW),
    (ArtifactFormat.TEXT, ArtifactFormat.BIN),
    # source/text -> json/yaml
    # (ArtifactFormat.SOURCE, ArtifactFormat.JSON),
    # (ArtifactFormat.SOURCE, ArtifactFormat.YAML),
    # (ArtifactFormat.TEXT, ArtifactFormat.JSON),
    # (ArtifactFormat.TEXT, ArtifactFormat.YAML),
])
def test_convert_artifacts_source_text_valid(from_, to):
    in_artifact = Artifact("foo.txt", content="foobar", fmt=from_)
    out_artifact = in_artifact.convert(to)
    out_artifact.validate()
    assert out_artifact.fmt == to
    # TODO": check for equality


@pytest.mark.parametrize("from_,to,raises", [
    (ArtifactFormat.SOURCE, ArtifactFormat.UNKNOWN, NotImplementedError),
    (ArtifactFormat.TEXT, ArtifactFormat.UNKNOWN, NotImplementedError),
    (ArtifactFormat.SOURCE, ArtifactFormat.MLF, NotImplementedError),
    (ArtifactFormat.TEXT, ArtifactFormat.MLF, NotImplementedError),
    # ...
])
def test_convert_artifacts_source_text_invalid(from_, to, raises):
    in_artifact = Artifact("foo.bar", content="foobar", fmt=from_)
    with pytest.raises(raises):
        _ = in_artifact.convert(to)
    # TODO: expect error


# TODO: MLF, MODEL, IMAGE, DATA, NUMPY, PARAMS, JSON, YAML, PATH, RAW, BIN, SHARED_OBJECT, ARCHIVE
