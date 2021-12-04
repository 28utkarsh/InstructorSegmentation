# Instructor Segmentation

## Project Objectives

1. To remove the monocolor background from instructor videos.
2. To develop a robust algorithm that works in real-time
3. The algorithm will work even when the background consists of real world scenes (curtains, windows, doors, etc)


## Setup Instructions

Run the following commands to download the DeepLab models and install the necessary dependencies.

```
pip install requirements.txt

wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
mkdir mobile_net_model && tar xvzf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz -C mobile_net_model
mkdir xception_model && tar xvzf deeplabv3_pascal_train_aug_2018_01_04.tar.gz
```

## Execution Instructions

Segmentation can be performed on videos by running one of the following commands:

```
# Fast 25 FPS model
python seg_contour.py input_file.mp4 model 3 --video

# Accurate Model
python seg_contour.py input_file.mp4 model 2 --video

# Hybrid Heuristics based approach
python seg_v2.py ../green.mp4 --output_fpath out.avi --model_type x
```

# Hackathon Information

This project was built in Bootstrap Paradox Hackathon organized by Blume Ventures. Also, our approach was considered in the top-5 list of all the ideas. Please click on this [link](https://skillenza.com/challenge/bootstrap-paradox) to know more about the event.