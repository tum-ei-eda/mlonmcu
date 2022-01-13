import sys

from .backend import TVMBackend
from mlonmcu.flow.backend import main


class TVMRTBackend(TVMBackend):

    shortname = "tvmrt"

    def generate_code(self):
        pass


if __name__ == "__main__":
    sys.exit(
        main(
            "tvmrt",
            TVMRTBackend,
            backend_features=FEATURES,
            backend_defaults=DEFAULT_CONFIG,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
