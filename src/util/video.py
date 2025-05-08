import json
import os
import glob
from typing import TypedDict, Optional

class CamConfig(TypedDict):
    front: str
    left: str
    right: str
def get_camera_config(config_folder: str = "assets/configs") -> Optional[CamConfig]:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    abs_config_folder = os.path.join(project_root, config_folder)

    for filename in glob.glob(os.path.join(os.path.normpath(abs_config_folder), '*.json')):
        with open(filename, 'r') as fp:
            config: CamConfig = json.load(fp)
            if any((os.path.exists(camera) and not os.path.isdir(camera) for camera in config.values())):
                return config
    return None
