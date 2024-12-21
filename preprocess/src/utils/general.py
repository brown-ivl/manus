import os
import logging
import shutil
import os
import src.utils.params as params_utils
import json
from typing import List

def create_dir(path: str):
    """Creates a directory also deleting previous one if it exissts"""
    logging.warning(f"Deleting files at {path}")
    try:
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path)


def get_chosen_frames(file_paths: List[str], chosen_list: str) -> List[str]:
    chosen = []
    with open(chosen_list, "r") as f:
        chosen_frames = set(json.load(f))
        
    for path in file_paths:
        if int(os.path.basename(path).split(".")[0]) in chosen_frames:
            chosen.append(path)
    return chosen