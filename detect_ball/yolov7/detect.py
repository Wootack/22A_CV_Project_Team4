import argparse
import sys

import torch
from numpy import ascontiguousarray

from .models.experimental import attempt_load
from .utils.datasets import letterbox
from .utils.general import check_img_size, non_max_suppression, scale_coords
from .utils.torch_utils import select_device

sys.path.insert(0, './detect_ball/yolov7/models')

def prepare_yolov7():
    imgsz = 640
    weights = './detect_ball/yolov7/models/yolov7-e6e.pt'
    # weights = 'models/yolov7-e6e.pt'
    # Initialize
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model, stride, device


def detect(model, stride, device, im0s):
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45
    half = device.type != 'cpu'
    img = letterbox(im0s, imgsz, stride)[0]
    img = img[:,:,::-1].transpose(2,0,1)
    img = ascontiguousarray(img)

    for _ in range(1):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        for i in range(3):
            model(img, augment=False)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                xywhs = []

                for *xyxy, conf, cls in reversed(det):
                    if cls != 32: continue
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    w = int(xyxy[2]) - x1
                    h = int(xyxy[3]) - y1
                    xywhs.append([x1, y1, w, h])
                return xywhs
            else: return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect()
