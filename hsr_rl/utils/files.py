import hsr_rl
from pathlib import Path

# get paths
def get_root_path():
    path = Path(hsr_rl.__path__[0]).resolve() / '..' / '..'
    return path


def get_urdf_path():
    path = get_root_path() / 'urdf'
    return path


def get_urdf_path():
    path = Path(hsr_rl.__path__[0]).resolve()/ 'models' / 'urdf'
    return path


def get_usd_path():
    path = Path(hsr_rl.__path__[0]).resolve()/ 'models' / 'usd'
    return path


def get_cfg_path():
    path = path = Path(hsr_rl.__path__[0]).resolve()/ 'cfg'
    return 