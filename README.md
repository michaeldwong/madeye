# MadEye

## Overview

This repository contains the simulation code described in the paper, "MadEye: Boosting Live Video Analytics Accuracy with Adaptive Camera Configurations". A preprint of our paper is available here: https://arxiv.org/pdf/2304.02101.pdf. The final version will be released soon.

## Installation

Install the requisite python packages

```python3 -m pip install -r requirements.txt```



## Quick start


First download all of the requisite files from here: https://www.dropbox.com/scl/fo/yu2lnitjxnrulydvfbn1k/h?rlkey=0ruexixzsay11c65w3iejcky0&dl=0

In ```config.yml```, modify ```saved_data_dir``` to the the location of these files, ```project_name``` to whichever dataset you are using (e.g., seattle-dt-1), and ```map_result_file```  to the path to the txt file containing all of the mAP results (e.g., MadEye/seattle-dt-1/seattle-dt-1-map.txt). Each sub-directory in the ```saved_data_dir``` corresponds to a different dataset.  Each subdirectory contains the mAP file and a directory called ```data``` which holds all of the data structures from the dataset.


To execute the simulation, simply run

```python3 main.py```




## Usage


The file ```config.yml``` contains all of the parameters for MadEye to run.


Here is a description of each of the parameters:

```faster_rcnn_weights``` -- path to the weights file for the EfficientDet model to mimic Faster RCNN
```yolov4_weights``` -- path to the weights file for the EfficientDet model to mimic YOLOv4
```tiny_yolov4_weights``` -- path to the weights file for the EfficientDet model to mimic Tiny YOLOv4
```ssd_voc_weights``` -- path to the weights file for the EfficientDet model to mimic SSd
```use_efficientdet``` -- True or False depending on whether you want to run the approximation models
```gpu``` -- The GPU device ID to run the models on. -1 to run on the CPU
```continual_learning``` -- True or False depending on whether you want to continually train the model
```num_frames_to_send``` -- Number of frames the camera can send to the server in 1 timestep
```num_frames_to_explore``` -- Number of different orientations the camera can visit
```save_to_pkl``` -- True or False depending on whether you want to save the data structures to .pkl files. When using a new dataset, this can save lots of time across executions.
```project_name``` -- Name of the project/video you're using. data structures will be stored/accessed using this name.
```rectlinear_dir``` -- Path to directory containing images. This is needed for continual learning and for getting mAP results, otherwise ignore
```inference_dir```: Path to directory containing backend model results. If using prior ```.pkl``` files, you can ignore this
```map_project_location```: Path to directory to mAP computation project (https://github.com/michaeldwong/mAP). This will update ```map_results_file```. If ```map_results_file``` is complete, you can ignore this.
```map_results_file```: Path to file containing mAP scores
```mot_results_dir```: Path to directory containing multi object tracking results. If using prior ```.pkl``` files, you can ignore this.
```saved_data_dir```: Directory containing the data structures in ```.pkl``` files. Thesea re the same files that are downloaded from DropBox. See ```main.py``` for how the expected 



