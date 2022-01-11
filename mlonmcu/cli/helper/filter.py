def filter_arg(arg):
    """TODO"""
    if not arg:
        return []
    return list(filter(None, arg.split(",")))
