def remove_config_prefix(config, prefix, skip=[]):
    def helper(key):
        return key.split(f"{prefix}.")[-1]

    return {
        helper(key): value
        for key, value in config.items()
        if f"{prefix}." in key and key not in skip
    }


def filter_config(config, prefix, defaults, required_keys):
    cfg = remove_config_prefix(config, prefix, skip=required_keys)
    for required in required_keys:
        value = None
        if required in cfg:
            value = cfg[required]
        elif required in config:
            value = config[required]
            cfg[required] = value
        assert value is not None, f"Required config key can not be None: {required}"

    for key in defaults:
        if key not in cfg:
            cfg[key] = defaults[key]

    for key in cfg:
        if key not in list(defaults.keys()) + required_keys:
            logger.warn("Backend received an unknown config key: %s", key)

    return cfg
