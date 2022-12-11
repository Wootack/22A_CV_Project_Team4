import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger
from easydict import EasyDict as edict

from video_track.ByteTrack.yolox.data.data_augment import preproc
from video_track.ByteTrack.yolox.exp import get_exp
from video_track.ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from video_track.ByteTrack.yolox.utils.visualize import plot_tracking
from video_track.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from video_track.ByteTrack.yolox.tracking_utils.timer import Timer
from mask_crowd.mask_crowd import mask_crowd

default_track_params = edict({'fps' : 30,
                'track_thresh' : 0.5,
                'track_buffer' : 30,
                'match_thresh' : 0.8,
                'aspect_ratio_thresh' : 1.6,
                'min_box_area' : 10,
                'mot20' : False})
SAVE_RESULT = True
VIDEOPATH = "./videos/2022-11-2217-13-37.mp4"


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer=None):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            if timer is not None: timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


def external_call(video, track_params=default_track_params):
    exp_file = "./video_track/ByteTrack/exps/example/mot/yolox_x_mix_det.py"
    exp = get_exp(exp_file, None)

    to_device = torch.device("cuda")

    model = exp.get_model().to(to_device)
    model.eval()

    ckpt_file = "./video_track/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar"
    ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    fp16 = True
    model = model.half()  # to FP16
    decoder = None

    predictor = Predictor(model, exp, decoder, to_device, fp16)
    return return_tracking(predictor, video, track_params)


def return_tracking(predictor, cap, track_params):
    tracker = BYTETracker(track_params, frame_rate=30)
    results = []
    frame_id = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            frame = mask_crowd(frame)
            outputs, img_info = predictor.inference(frame)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], predictor.test_size)
                online_tlwhs = []
                online_ids = []
                # online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > track_params.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > track_params.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        # online_scores.append(t.score)
                # results.append([frame_id, online_ids, online_tlwhs, online_scores])
                results.append([online_ids, online_tlwhs])
            else: results.append([[], [], []])
        else:
            break
        frame_id += 1
    return results


def imageflow_demo(predictor, vis_folder, current_time):
    cap = cv2.VideoCapture(VIDEOPATH)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, VIDEOPATH.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(default_track_params, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], predictor.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > default_track_params.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > default_track_params.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if SAVE_RESULT:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if SAVE_RESULT:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main():
    exp_file = "exps/example/mot/yolox_x_mix_det.py"
    exp = get_exp(exp_file, None)

    output_dir = osp.join(exp.output_dir, exp.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    if SAVE_RESULT:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    to_device = torch.device("cuda")

    # logger.info("Args: {}".format(args))

    model = exp.get_model().to(to_device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
 

    ckpt_file = "pretrained/bytetrack_x_mot17.pth.tar"
    ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)

    fp16 = True

    model = model.half()  # to FP16

    decoder = None

    predictor = Predictor(model, exp, decoder, to_device, fp16)
    current_time = time.localtime()
    imageflow_demo(predictor, vis_folder, current_time)


if __name__ == "__main__":

    main()
