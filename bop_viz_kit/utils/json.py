import numpy as np
import json
import ruamel.yaml as yaml


def load_json(path):
    with open(path, "r") as f:
        info = yaml.load(f, Loader=yaml.CLoader)
    return info


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)
