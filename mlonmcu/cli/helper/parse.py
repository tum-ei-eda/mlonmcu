def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    assert "=" in s, "Not a key-value pair"
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    assert len(key) > 0, "Empty key"
    assert len(items) > 1, "Not enough items"
    if len(items) > 1:
        # rejoin the rest:
        value = "=".join(items[1:])
    return (key, value)


def parse_vars(items):  # TODO: this needs to be used in other subcommands as well?
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            if len(item) > 0:
                key, value = parse_var(item)
                d[key] = value
    return d


def extract_features(args):
    if args.feature:
        return args.feature
    else:
        return []


def extract_config(args):
    if args.config:
        configs = sum(args.config, [])
        configs = parse_vars(configs)
    else:
        configs = {}
    return configs
