"""MLonMCU platform submodule"""

from .mlif import MlifPlatform
from .espidf import EspIdfPlatform

# from .arduino import ArduinoPlatform

SUPPORTED_PLATFORMS = {
    "mlif": MlifPlatform,
    "espidf": EspIdfPlatform,
    # "arduino": ArduinoPlatform,
}
