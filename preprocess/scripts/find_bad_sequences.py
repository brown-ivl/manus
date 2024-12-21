import sys
sys.path.append(".")

import os
import json
import argparse
import glob
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-r", required=True, type=str)
    parser.add_argument("--seq_dir", "-s", required=True, type=str)
    parser.add_argument("--filter_ratio", required=False, type=int, default=0.6)
    parser.add_argument("--diff_ratio",  required=False, type=int, default=0.8)
    
    args = parser.parse_args()
    
    base_path = os.path.join(args.root_dir, args.seq_dir)
    
    all_seqs = glob.glob(os.path.join(base_path, "*"))
    
    faulty_seqs = [] 
    for seq in all_seqs:
        view = glob.glob(os.path.join(seq, "pose_dumps", "keypoints_2d", '*'))[0]
        kpts2d = natsorted(glob.glob(os.path.join(view, "*.json")))
        chosen_frames_path = os.path.join(seq, "pose_dumps", "chosen_frames.json")
        with open(chosen_frames_path, "r") as f:
            chosen_frames = json.load(f)
            
        # if (len(chosen_frames) / len(kpts2d)) > args.filter_ratio:
        if (chosen_frames[-1] / int(kpts2d[-1].split("/")[-1].split(".")[0])) > args.diff_ratio:
            continue
        else:
            faulty_seqs.append(seq)
    
    breakpoint()
    faulty_seqs_path = os.path.join(args.root_dir, f"faulty_{args.seq_dir}.txt") 
    if os.path.exists(faulty_seqs_path):
        faulty_seqs_path = faulty_seqs_path.replace(".txt", "_1.txt")
        if os.path.exists(faulty_seqs_path):
            faulty_seqs_path = faulty_seqs_path.replace("_1.txt", "_2.txt")
            
    
    print("Writing Faulty Sequences to: ", faulty_seqs_path) 
    with open(faulty_seqs_path, "w") as f: 
        for seq in faulty_seqs:
            seq = seq.split("/")[-1]
            f.write(seq + "\n")
        
if __name__ == '__main__':
    main()