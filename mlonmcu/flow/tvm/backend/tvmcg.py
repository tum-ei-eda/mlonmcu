import sys

from .backend import TVMBackend
from mlonmcu.flow.backend import main


class TVMCGBackend(TVMBackend):

    shortname = "tvmcg"

    def generate_code(self):
        pass


if __name__ == "__main__":
    sys.exit(
        main(
            "tvmcg",
            TVMCGBackend,
            backend_features=FEATURES,
            backend_defaults=DEFAULT_CONFIG,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
