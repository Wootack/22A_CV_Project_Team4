# 22A_CV_Project_Team4
2022 Autumn Computer Vision course Term Project - Team 4; Offside Detection

Follow these steps for initial configuration.
```bash
conda env create -n cv_Offside_Detection -f py37environment.yml
conda activate cv_Offside_Detection
cd ./video_track/ByteTrack
python setup.py develop
cd ../..
```
Download pre-trained model weight values from [this link](https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5) and upload it to ```./video_track/ByteTrack/pretrained/``` directory.
Also download ```coco.names```, ```yolov3.cfg```, and ```yolov3.weights``` from [this link](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html) and upload them to ```./darknet/cfg/``` directory.


Upload input mp4 video to ```./data``` directory.