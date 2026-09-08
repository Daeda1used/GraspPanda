"""Download every required role for a method/camera, with recorded checksums."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasppanda.weights import fetch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method')
    parser.add_argument('--camera', choices=['realsense','kinect'], default='realsense')
    args=parser.parse_args()
    print(fetch(args.method,args.camera))

if __name__=='__main__': main()
