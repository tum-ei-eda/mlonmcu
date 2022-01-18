import sys

from ..tvm_flow import get_parser
from ..framework import COMMON_TVM_CONFIG

from .backend import TVMBackend
from mlonmcu.flow.backend import main

FEATURES = [
    "debug_arena",
    "unpacked_api",
    "autotuned",
]  # TODO COMMON_TVM_FEATURES(autotuned) -> Framework?

# COMMON_TVM_CONFIG = {}

DEFAULT_CONFIG = {
    **COMMON_TVM_CONFIG,
    **{
        "arena_size": -1,  # Determined automatically
        # "unpacked_api": False,  # Actually a feature, so ommit here?
        "alignment_bytes": 4,
        "tuning_log_file": "tuning.log.txt",  # TODO: update and share between backends
    },
}


class TVMAOTBackend(TVMBackend):

    shortname = "tvmaot"

    def __init__(self, features=None, config=None, context=None):
        super().__init__(features=features, config=config, context=context)
        # self.unpacked_api, self.get_debug_arena = self.resolve_features()
        # self.arena_size, self.alignment_bytes = self.resolve_config()
        # self.target = self.get_target()

    # def resolve_features(self):
    #     unpacked_api = False
    #     debug_arena = False
    #     for feature in self.features:
    #         if feature.name == "unpacked_api":
    #             unpacked_api = True
    #         elif feature.name == "debug_arena":
    #             debug_arena = True
    #     return (unpacked_api, debug_arena)

    # def resolve_config(self):
    #     arena_size = DEFAULT_CONFIG["arena_size"]
    #     alignment_bytes = DEFAULT_CONFIG["alignment_bytes"]
    #     for key, value in self.config:
    #         if key.split(".")[-1] == "arena_size":
    #             arena_size = int(value)
    #         if key.split(".")[-1] == "alignment_bytes":
    #             alignment_bytes = int(value)
    #     return (arena_size, alignment_bytes)

    # def get_target_str(self):
    #     target_str = super().get_target_str(self)
    #     target_str += " --link-params"
    #     target_str += " --executor=aot"
    #     target_str += " --workspace-byte-alignment={}".format(self.alignment_bytes)
    #     target_str += " --unpacked-api={}".format(int(self.unpacked_api))
    #     target_str += " --interface-api={}".format(
    #         "c" if self.unpacked_api else "packed"
    #     )
    #     return target_str

    def generate_code(self):
        # assert self.target
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
