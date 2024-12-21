from argparse import ArgumentParser

def add_common_args(parser: ArgumentParser):
    parser.add_argument("--root_dir", "-r", required=True, type=str)
    parser.add_argument("--input_type", "-t", default="video", choices=["video", "image"], help="Whether the input is a video or set of images")
    parser.add_argument("--seq_path", "-s", type=str, required=True)
    parser.add_argument("--handedness", choices=["left", "right"], default="right", type=str)
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--start", default=0, type=int, help="Start frame")
    parser.add_argument("--end", default=-1, type=int, help="End frame")