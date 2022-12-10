import os
import argparse
import pickle

import cv2
import numpy as np

from detect_ball import db_yolo
from detect_human import dh_yolo, dh_hsv
from video_track import vt_bytetrack
from team_cluster import team_cluster


datapath = './data/'
ATTACKER = 'red'
RECEIVED = 'touch'
INIT_BALL = np.array([879, 287])


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
    # Single File
    if os.path.isfile(input_path):
        video1 = cv2.VideoCapture(input_path)
        end_to_end_pipeline(video1, ATTACKER, RECEIVED, INIT_BALL, args)

    # All files in Directory
    # The directory should include 'state.pickle', which specifies ...
    # ... attacker, receiver's touch, and initial bounding boxes.
    elif os.path.isdir(input_path):
        filelist = os.listdir(input_path)
        filelist.sort()
        with open(os.path.join(input_path, 'state.pickle'), 'rb') as fp:
            attack_teams, receive_touch, init_balls = pickle.load(fp)
        for i, iv in enumerate(filelist):
            if not iv.endswith('.mp4'): continue
            input_video = os.path.join(input_path, iv)
            video = cv2.VideoCapture(input_video)
            end_to_end_pipeline(video, attack_teams[i], receive_touch[i],
                    init_balls[i], args)


def end_to_end_pipeline(video1, attacker, received, init_ball, args):
    # BALL DETECTION
    ball_xywh_array = db_yolo.db_yolo(video1, init_ball)

    # HUMAN DETECTION
    if args.tracking_method=='yolo':
        raise NotImplementedError()
        human_tids_tlwhs_list = dh_yolo.dh_yolo(video1) # return of dh_yolo should be modified
    elif args.tracking_method=='bytetrack':
        human_tids_tlwhs_list = vt_bytetrack.vt_bytetrack(video1, args.save_video)
    elif args.tracking_method=='hsv':
        human_tids_tlwhs_list = dh_hsv.dh_hsv(video1, args.save_video)

    with open('./temp.pickle', 'wb') as fp:
        pickle.dump(human_tids_tlwhs_list, fp, pickle.HIGHEST_PROTOCOL)

    # TEAM CLUSTERING
    red_tids, blue_tids = team_cluster.team_cluster(human_tids_tlwhs_list, video1)
    
    print('DONE')


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)
