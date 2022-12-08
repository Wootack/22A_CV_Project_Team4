import os
import argparse

import cv2

from detect_human import dh_yolo, dh_hsv
from video_track import vt_bytetrack
from team_cluster import team_cluster


datapath = './data/'


def make_parser():
    parser = argparse.ArgumentParser("Offside Detecting System")
    parser.add_argument("-iv", "--input-video", default="Game.mp4",
            help="Name of input video file. Relative path in ./data directory. mp4 file is recommended.")
    parser.add_argument("-tm", "--tracking-method", default="bytetrack", choices=['bytetrack', 'yolo', 'hsv'],
            help="Determine player tracking method")
    parser.add_argument("-sv", "--save-video", default=True, action="store_true", help="Save output video")
    return parser


def main(args):
    os.makedirs('./results', exist_ok=True)
    print('Hi')
    input_path = os.path.join(datapath, args.input_video)
    if not os.path.exists(input_path):
        raise Exception('Input Video not existing.')
    print('Detecting Offside from', input_path)
    video1 = cv2.VideoCapture(input_path)

    # Current Issue : Mixed usage of term; detection and tracking.
    # Q. Should we separate them?
    if args.tracking_method=='yolo':
        id_tlwh_score_per_frame = dh_yolo.dh_yolo(video1) # return of dh_yolo should be modified
    elif args.tracking_method=='bytetrack':
        id_tlwh_score_per_frame = vt_bytetrack.vt_bytetrack(video1, args.save_video)
    elif args.tracking_method=='hsv':
        # TODO
        id_tlwh_score_per_frame = dh_hsv.dh_hsv(video1, args.save_video)

    team_cluster.team_cluster(id_tlwh_score_per_frame, args.save_video, video1)
    
    print('DONE')


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)
