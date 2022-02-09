import sys
import os
import tempfile
import logging
from pathlib import Path
from .backend import TFLiteBackend
import mlonmcu.setup.utils as utils
from mlonmcu.flow.backend import main
from mlonmcu.logging import get_logger
from mlonmcu.artifact import Artifact, ArtifactFormat

logger = get_logger()


class TFLMCBackend(TFLiteBackend):

    name = "tflmc"

    FEATURES = ["debug_arena"]

    DEFAULTS = {
        **TFLiteBackend.DEFAULTS,
        "custom_ops": [],
        "registrations": {},
        "debug_arena": False,
    }

    REQUIRED = TFLiteBackend.REQUIRED + ["tflmc.exe"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(features=features, config=config, context=context)
        self.model_data = None
        self.prefix = "model"  # Without the _
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data

    def generate_code(self, verbose=False):
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
            utils.exec_getout(tflmc_exe, *args, live=verbose)
            files = [f for f in os.listdir(tmpdirname) if os.path.isfile(os.path.join(tmpdirname, f))]
            # TODO: ensure that main file is processed first
            for filename in files:
                with open(Path(tmpdirname) / filename, "r") as handle:
                    content = handle.read()
                    artifacts.append(Artifact(filename, content=content, fmt=ArtifactFormat.SOURCE))

        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TFLMCBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
