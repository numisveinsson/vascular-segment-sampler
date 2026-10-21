import os
import yaml
import csv

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_NAME = "default_mm"
# YAML `KEY:` with no value loads as None; treat these as empty lists.
_EMPTY_LIST_KEYS = ("TEST_CASES", "BAD_CASES")


def _coerce_empty_lists(cfg):
    """Replace null list-valued keys with ``[]`` so callers can use ``len()`` / ``in``."""
    for key in _EMPTY_LIST_KEYS:
        if cfg.get(key) is None:
            cfg[key] = []
    return cfg


def resolve_config_path(config):
    """Resolve a config name or path to a YAML file.

    Bare names look in ``./config/`` (experiment overlays) then the packaged
    ``vascular_segment_sampler/config/`` (library defaults: ``default_mm``,
    ``default_cm``, ``test.yml``). Existing file paths
    are used as-is.
    """
    config_path = str(config)
    if os.path.isfile(config_path):
        return config_path

    if config_path.endswith((".yaml", ".yml")):
        candidate = config_path
    else:
        candidate = config_path + ".yaml"
    basename = os.path.basename(candidate)
    cwd_cfg = os.path.join("config", basename)
    pkg_cfg = os.path.join(_PKG_DIR, "config", basename)

    if os.path.isfile(cwd_cfg):
        return cwd_cfg
    if os.path.isfile(pkg_cfg):
        return pkg_cfg
    if os.path.isfile(candidate):
        return candidate
    return cwd_cfg if os.path.isdir("config") else pkg_cfg


def load_default_config():
    """Load packaged ``default_mm.yaml``, or ``{}`` if it is missing."""
    path = resolve_config_path(DEFAULT_CONFIG_NAME)
    if not os.path.isfile(path):
        return {}
    loaded = load_yaml(path)
    return _coerce_empty_lists(dict(loaded) if loaded else {})


def load_config(config):
    """Load a YAML config, overlaying it on packaged ``default_mm.yaml``.

    ``config`` may be a dict, a file path, or a bare name (repo-root ``config/``
    overlay or packaged library default). List values (e.g. ``ANATOMY``,
    ``TEST_CASES``) are replaced, not merged.
    """
    if isinstance(config, dict):
        overlay = dict(config)
    else:
        default_path = os.path.abspath(resolve_config_path(DEFAULT_CONFIG_NAME))
        config_path = resolve_config_path(config)
        overlay = load_yaml(config_path)
        if overlay is None:
            overlay = {}
        else:
            overlay = dict(overlay)
        if os.path.isfile(default_path) and os.path.abspath(config_path) == default_path:
            return _coerce_empty_lists(overlay)

    defaults = load_default_config()
    return _coerce_empty_lists({**defaults, **overlay})


def prompt_continue(message="Wish to continue? [y/n]: "):
    """Prompt until a valid y/n answer is provided."""
    while True:
        answer = input(message).strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("Please answer with 'y' or 'n'.")


def read_lines(fn):
    f = open(fn,'r').readlines()
    f = [s.replace('\n','') for s in f]
    return f


def load_yaml(fn):
    """loads a yaml file into a dict"""
    with open(fn,'r') as file_:
        try:
            return yaml.load(file_, Loader=yaml.loader.SafeLoader)
        except RuntimeError as e:
            print("failed to load yaml fille {}, {}\n".format(fn,e))

def save_yaml(fn, data):
    with open(fn,'w') as file_:
        yaml.dump(data,file_, default_flow_style=False)

def write_csv(filename,dict):
    with open(filename,'w') as f:
        w = csv.DictWriter(f,dict.keys())
        w.writeheader()
        w.writerow(dict)

def read_csv(filename):
    with open(filename,'r') as f:
        w = csv.DictReader(f)
        d = [row for row in w]
        if len(d)==1:
            return d[0]
        else:
            return d
