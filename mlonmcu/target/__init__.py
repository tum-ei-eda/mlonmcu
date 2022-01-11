"""MLonMCU target submodule"""

from .target import Target

SUPPORTED_TARGETS = {
    "etiss/pulpino": Target("etiss/pulpino"),
    "host": Target("host"),
}  # TODO
