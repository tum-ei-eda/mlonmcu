"""Artifacts defintions internally used to refer to intermediate results."""

import os
from enum import Enum
from pathlib import Path

from mlonmcu.setup import utils

# class ModelLibraryFormatPlus:
#     pass


class ArtifactFormat(Enum):  # TODO: ArtifactType, ArtifactKind?
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


class Artifact:
    """Artifact type."""

    def __init__(
        self,
        name,
        content=None,
        data=None,
        raw=None,
        fmt=ArtifactFormat.UNKNOWN,
        archive=False,
        optional=False,
    ):
        # TODO: Allow to store filenames as well as raw data
        self.name = name
        self.content = content
        self.data = data
        self.raw = raw
        self.fmt = fmt
        self.archive = archive
        self.optional = optional
        self.validate()

    def validate(self):
        if self.fmt in [ArtifactFormat.TEXT, ArtifactFormat.SOURCE]:
            assert self.content is not None
        elif self.fmt in [ArtifactFormat.RAW, ArtifactFormat.BIN]:
            assert self.raw is not None
        elif self.fmt in [ArtifactFormat.MLF]:
            assert self.raw is not None  # TODO: load it via tvm?
        elif self.fmt in [ArtifactFormat.PATH]:
            assert self.path is not None
        else:
            raise NotImplementedError

    def export(self, dest, extract=False):
        filename = Path(dest) / self.name
        if self.fmt in [ArtifactFormat.TEXT, ArtifactFormat.SOURCE]:
            assert (
                not extract
            ), "extract option is only available for ArtifactFormat.MLF"
            with open(filename, "w") as handle:
                handle.write(self.content)
        elif self.fmt in [ArtifactFormat.RAW, ArtifactFormat.BIN]:
            assert (
                not extract
            ), "extract option is only available for ArtifactFormat.MLF"
            with open(filename, "wb") as handle:
                handle.write(self.data)
        elif self.fmt in [ArtifactFormat.MLF]:
            with open(filename, "wb") as handle:
                handle.write(self.raw)
            if extract:
                utils.extract(filename, dest)
                os.remove(filename)
        elif self.fmt in [ArtifactFormat.PATH]:
            assert (
                not extract
            ), "extract option is only available for ArtifactFormat.MLF"
            utils.copy(self.path, filename)
        else:
            raise NotImplementedError

    def print_summary(self):
        print("Format:", self.fmt)
        print("Optional: ", self.optional)
        if self.fmt in [ArtifactFormat.TEXT, ArtifactFormat.SOURCE]:
            print("Content:")
            print(self.content)
        elif self.fmt in [ArtifactFormat.RAW, ArtifactFormat.BIN]:
            print(f"Data Size: {len(self.raw)}B")
        elif self.fmt in [ArtifactFormat.MLF]:
            print(f"Archive Size: {len(self.raw)}B")
        elif self.fmt in [ArtifactFormat.Path]:
            print(f"File Location: {self.path}")
        else:
            raise NotImplementedError
