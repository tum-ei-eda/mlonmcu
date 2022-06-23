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
"""Artifacts defintions internally used to refer to intermediate results."""

from enum import Enum
from pathlib import Path

from mlonmcu.setup import utils

# TODO: offer pack/unpack/flatten methods for mlf
# TODO: implement restore methods
# TODO: decide if inheritance based scheme would fit better
# TODO: add artifact flags and lookup utility to find best match


class ArtifactFormat(Enum):  # TODO: ArtifactType, ArtifactKind?
    """Enumeration of artifact types."""

    UNKNOWN = 0
    SOURCE = 1
    TEXT = 2
    MLF = 3
    MODEL = 4
    IMAGE = 5
    DATA = 6
    NUMPY = 7
    PARAMS = 8
    JSON = 9  # ?
    PATH = 10  # NOT A DIRECTORY?
    RAW = 11
    BIN = 11
    SHARED_OBJECT = 12  # Here: the parent tar archive


def lookup_artifacts(artifacts, name=None, fmt=None, flags=None, first_only=False):
    """Utility to get a matching artifact for a given set of properties."""
    matches = []
    # Warning: if neither name, fmt nor flags is provided, the first artifact (multiple=False)
    # or all (multiple=True) are returned
    for artifact in artifacts:
        valid = True
        if name is not None and artifact.name != name:
            valid = False
        if fmt is not None and artifact.fmt != fmt:
            valid = False
        if flags is not None and not all(
            flag in artifact.flags for flag in flags
        ):  # A valid artifact may have more flags than specified
            valid = False
        if valid:
            matches.append(artifact)
    if len(matches) > 0 and first_only:
        matches = [matches[0]]
    return matches


class Artifact:
    """Artifact type."""

    def __init__(
        self,
        name,
        content=None,
        path=None,
        data=None,
        raw=None,
        fmt=ArtifactFormat.UNKNOWN,
        flags=None,
        archive=False,
        optional=False,
    ):
        # TODO: Allow to store filenames as well as raw data
        self.name = name
        # TODO: too many attributes...
        self.content = content
        self.path = path
        self.data = data
        self.raw = raw
        self.fmt = fmt
        self.flags = flags if flags is not None else {}
        self.archive = archive
        self.optional = optional
        self.validate()

    def __repr__(self):
        return f"Artifact({self.name}, fmt={self.fmt}, flags={self.flags})"

    @property
    def exported(self):
        """Returns true if the artifact was writtem to disk."""
        return bool(self.path is not None)

    def validate(self):
        """Checker for artifact attributes for the given format."""
        if self.fmt in [ArtifactFormat.TEXT, ArtifactFormat.SOURCE]:
            assert self.content is not None
        elif self.fmt in [ArtifactFormat.RAW, ArtifactFormat.BIN]:
            assert self.raw is not None
        elif self.fmt in [ArtifactFormat.MLF, ArtifactFormat.SHARED_OBJECT]:
            assert self.raw is not None
        elif self.fmt in [ArtifactFormat.PATH]:
            assert self.path is not None
        else:
            raise NotImplementedError

    def export(self, dest, extract=False):
        """Export the artifact to a given path (file or directory) and update its path.

        Arguments
        ---------
        dest : str
            Path of the destination.
        extract : bool
            If archive: extract to destination.

        """
        filename = Path(dest) / self.name
        if self.fmt in [ArtifactFormat.TEXT, ArtifactFormat.SOURCE]:
            assert not extract, "extract option is only available for ArtifactFormat.MLF"
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(self.content)
        elif self.fmt in [ArtifactFormat.RAW, ArtifactFormat.BIN]:
            assert not extract, "extract option is only available for ArtifactFormat.MLF"
            with open(filename, "wb") as handle:
                handle.write(self.raw)
        elif self.fmt in [ArtifactFormat.MLF, ArtifactFormat.SHARED_OBJECT]:
            with open(filename, "wb") as handle:
                handle.write(self.raw)
            if extract:
                utils.extract(filename, dest)
                # os.remove(filename)
        elif self.fmt in [ArtifactFormat.PATH]:
            assert not extract, "extract option is only available for ArtifactFormat.MLF"
            utils.copy(self.path, filename)
        else:
            raise NotImplementedError
        self.path = filename if self.path is None else self.path

    def print_summary(self):
        """Utility to print information about an artifact to the cmdline."""
        print("Format:", self.fmt)
        print("Optional: ", self.optional)
        if self.fmt in [ArtifactFormat.TEXT, ArtifactFormat.SOURCE]:
            print("Content:")
            print(self.content)
        elif self.fmt in [ArtifactFormat.RAW, ArtifactFormat.BIN]:
            print(f"Data Size: {len(self.raw)}B")
        elif self.fmt in [ArtifactFormat.MLF, ArtifactFormat.SHARED_OBJECT]:
            print(f"Archive Size: {len(self.raw)}B")
        elif self.fmt in [ArtifactFormat.PATH]:
            print(f"File Location: {self.path}")
        else:
            raise NotImplementedError
