import sys


def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0


def ask_user(text, default: bool, yes_keys=["y", "j"], no_keys=["n"], interactive=True):
    assert len(yes_keys) > 0 and len(no_keys) > 0
    if not interactive:
        return default
    if default:
        suffix = " [{}/{}] ".format(yes_keys[0].upper(), no_keys[0].lower())
    else:
        suffix = " [{}/{}] ".format(yes_keys[0].lower(), no_keys[0].upper())
    message = text + suffix
    answer = input(message)
    if default:
        return answer.lower() not in no_keys and answer.upper() not in no_keys
    else:
        return not (answer.lower() not in yes_keys and answer.upper() not in yes_keys)


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    """Detects if the current python interpreter is from a virtual environment."""
    return get_base_prefix_compat() != sys.prefix
