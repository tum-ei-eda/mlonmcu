import sys

from ..tvm_flow import get_parser
from ..framework import COMMON_TVM_CONFIG

from .backend import TVMBackend
from mlonmcu.flow.backend import main

FEATURES = ["debug_arena"]

# COMMON_TVM_CONFIG = {}

DEFAULT_CONFIG = {
    **COMMON_TVM_CONFIG,
    **{
        "arena_size": -1,  # Determined automatically
        "unpacked_api": False,
        "workspace-byte-alignment": 4,
    },
}


class TVMAOTBackend(TVMBackend):

    shortname = "tvmaot"

    def generate_code(self):
        pass


if __name__ == "__main__":
    sys.exit(
        main(
            "tvmaot",
            TVMAOTBackend,
            backend_features=FEATURES,
            backend_defaults=DEFAULT_CONFIG,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
