from enum import Enum


class FeatureType(Enum):
    """Enumeration for Feature types."""

    OTHER = 0
    SETUP = 1
    FRONTEND = 2
    FRAMEWORK = 3
    BACKEND = 4
    TARGET = 5
    # COMPILE = 6
    PLATFORM = 6
    RUN = 7
