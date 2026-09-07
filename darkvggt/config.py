import os

import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "darkvggt.yaml"
)


def load_model_kwargs(path=None):
    with open(path or CONFIG_PATH) as handle:
        kwargs = yaml.safe_load(handle)
    if not isinstance(kwargs, dict):
        raise SystemExit(f"{path or CONFIG_PATH} does not parse to a mapping")
    return kwargs
