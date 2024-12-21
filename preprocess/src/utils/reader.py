import cv2
import os
import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple

import ipdb

class Reader():
    iterator = []

    def __init__(
            self, inp_type: str, path: str, undistort: bool=False
        ):
        self.type = inp_type

        self.frame_count = int(1e9)
        if self.type == "video":
            self.streams = {}
            self.vids = natsorted(glob(f"{path}/*/*.avi"))
            self.init_videos()
        else:
            pass

        # Sanity checks
        assert self.frame_count > 0, "No frames found"

        self.cur_frame = 0
    
    def _get_next_frame(self) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        if self.cur_frame+1 == self.frame_count:
            return {}

        self.cur_frame += 1

        frames = {}
        for cam_name, cam_cap in self.streams.items():
            suc, frame = cam_cap.read()
            if not suc:
                raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")
            frames[cam_name] = frame
        
        return frames

    def reinit(self):
        """ Reinitialize the reader """
        if self.type == "video":
            self.release_videos()
            self.init_videos()

        self.cur_frame = 0

    def init_videos(self):
        """ Create video captures for each video """
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)
            frame_count = int(ffmpeg.probe(vid, cmd="ffprobe")["streams"][0]["nb_frames"])
            self.frame_count = min(self.frame_count, frame_count)
            cam_name = os.path.basename(vid).split(".")[0]
            self.streams[cam_name] = cap

    def release_videos(self):
        for cap in self.streams.values():
            cap.release()
    
    def __call__(self, frames: Iterable[int]=[]):
        # Sort the frames so that we access them in order
        frames = sorted(frames)
        self.iterator = frames

        for frame_idx in frames:
            # Skip through non needed frames
            while self.cur_frame < frame_idx:
                frame = self._get_next_frame()
                if not frame:
                    break

            frame = self._get_next_frame()
            if not frame:
                break
            
            yield frame, self.cur_frame - 1

        # Reinitialize the videos
        self.reinit()

if __name__ == "__main__":
    reader = Reader("video", "/hdd_data/common/BRICS/hands/peisen/actions/abduction_adduction/", 5, 16, 3)
    for i in range(len(reader)):
        frames, frame_num = reader.get_frames()
        print(frame_num)
