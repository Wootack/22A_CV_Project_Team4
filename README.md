# 22A_CV_Project_Team4
2022 Autumn Computer Vision course Term Project - Team 4; Offside Detection

Follow these steps for initial configuration.
```bash
conda create -y --no-default-packages -n cv-pr python=3.9.15
conda activate cv-pr
conda env update --name cv-pr --file environment.yaml --prune
cd ./video_track/ByteTrack
python setup.py develop
cd ../..
```
Download pre-trained model weight values from [this link](https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5) and upload it to '''./video_track/ByteTrack/pretrained/''' directory.
Upload input mp4 video to '''./data''' directory.

- Use ```conda install``` command to install packages. Use ```pip install``` only if it unavoidable.
- After package installation, enter ```conda env export > environment.yaml``` to update environment.yaml file.
- After pulling or fetching, update environment.
- You may need to change prefix of environment.yaml file.

Codes for ByteTrack algorithm are borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack) and modified for our purpose. Thanks for their wonderful work.