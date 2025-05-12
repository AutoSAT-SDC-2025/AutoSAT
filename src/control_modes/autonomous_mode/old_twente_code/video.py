import json
import os
import glob
from typing import TypedDict, Optional

class CamConfig(TypedDict):
    front: str
    left: str
    right: str

def get_camera_config(config_folder: str = "configs") -> Optional[CamConfig]:
    """
    Select a valid camera configuration from the config_folder and return it.

    Returns None if no valid configuration could be found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print('script_dir', script_dir)
    abs_config_folder = os.path.join(script_dir, config_folder)
    print('abs_config_folder', abs_config_folder)
    for filename in glob.glob(os.path.join(os.path.normpath(abs_config_folder), '*.json')):
        with open(filename, 'r') as fp:
            config: CamConfig = json.load(fp)
            if any((os.path.exists(camera) and not os.path.isdir(camera) for camera in config.values())):
                return config
    return None