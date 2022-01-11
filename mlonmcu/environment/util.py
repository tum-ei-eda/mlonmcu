import sys


def validate_name(name):
    # TODO: regex for valid names without spaces etc
    return True


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return (
        getattr(sys, "base_prefix", None)
        or getattr(sys, "real_prefix", None)
        or sys.prefix
    )


def in_virtualenv():
    """Detects if the current python interpreter is from a virtual environment."""
    return get_base_prefix_compat() != sys.prefix
