"""Plugin for 3rd party components"""

import os
import sys
from mlonmcu.logging import get_logger
from mlonmcu.target._target import register_target

logger = get_logger()
logger.warning("Executing Plugin: %s", __name__)

# Allow relative imports
sys.path.insert(0, os.path.dirname(__file__))
from _setup.tasks.abc import *  # noqa: E402,F401,F403
from _target.riscv.abc import ABCTarget  # noqa: E402


register_target("abc", ABCTarget)
