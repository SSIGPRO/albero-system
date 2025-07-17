import argparse
import configparser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def str2tuple(v):
    try:
        # Accepts formats like "1,2", "(1,2)", "1, 2", etc.
        v = v.strip("()")
        items = [float(x) if '.' in x else int(x) for x in v.split(",")]
        return tuple(items)
    except Exception:
        raise argparse.ArgumentTypeError("Tuple value expected in format '1,2' or '(1,2)'")


def get_parser_from_dict(config_dict):
    parser = argparse.ArgumentParser(description="Script with argparse + dict + config.ini")
    for key, val in config_dict.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=str2bool, nargs='?', const=True, default=None)
        elif isinstance(val, tuple):
            parser.add_argument(f"--{key}", type=str2tuple, default=None)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=None)
    return parser


def update_config_from_args(config_dict, args):
    for key in config_dict:
        val = getattr(args, key)
        if val is not None:
            config_dict[key] = val
    return config_dict


def save_config_to_ini(config_dict, path):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {k: str(v) for k, v in config_dict.items()}
    with open(path, 'w') as configfile:
        config.write(configfile)