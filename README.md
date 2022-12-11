# 22A_CV_Project_Team4
2022 Autumn Computer Vision course Term Project - Team 4; Offside Detection

Follow these steps for initial configuration.
```bash
conda env create -n cv_Offside_Detection -f py37environment.yml
conda activate cv_Offside_Detection
cd ./video_track/ByteTrack
python setup.py develop
cd ../..
python main.py -iv test.mp4
```
All the files to run the code: download from [this link](https://drive.google.com/drive/folders/1hTEujaJH1knKA6AQ_BgsIJ9vV6QLswsV?usp=share_link)
- Upload ```bytetrack_x_mot17.pth.tar``` to ```./video_track/ByteTrack/pretrained/``` directory.
- Upload ```coco.names```, ```yolov3.cfg```, and ```yolov3.weights``` to ```./darknet/cfg/``` directory.
- Upload input mp4 video to ```./data``` directory.