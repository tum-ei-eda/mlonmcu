from .platform import register_platform
from .mlif import MlifPlatform
from .espidf import EspIdfPlatform

# from .arduino import ArduinoPlatform

register_platform("mlif", MlifPlatform)
register_platform("espidf", EspIdfPlatform)
