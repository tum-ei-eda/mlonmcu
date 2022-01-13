import sys
from .backend import TFLiteBackend
from mlonmcu.flow.backend import main

FEATURES = ["debug_arena"]

DEFAULT_CONFIG = {
    "registrations": {},
}


class TFLMCBackend(TFLiteBackend):

    shortname = "tflmc"

    def generate_code(self):
        pass


if __name__ == "__main__":
    sys.exit(
        main(
            "tflmc",
            TFLMCBackend,
            backend_features=FEATURES,
            backend_defaults=DEFAULT_CONFIG,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
