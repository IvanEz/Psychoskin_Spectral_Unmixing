def load_config(config_file):
    import configparser

    config = configparser.ConfigParser()
    config.read(config_file)
    
    return {
        "data_dir": config.get("DEFAULT", "data_dir", fallback="data"),
        "prefix": config.get("DEFAULT", "prefix", fallback="sample"),
        "strategy": config.get("DEFAULT", "strategy", fallback="MinMax"),
        "n_unknown": config.getint("DEFAULT", "n_unknown", fallback=2)
    }