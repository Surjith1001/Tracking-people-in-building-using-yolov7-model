# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Getting Started
    
    To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version   for you.
    
## Conda (recommended)
       CPU
        conda create -n yolov7_tracking python=3.11(your puthon version)
        conda activate yolov4_tracking

       GPU
        conda create -n yolov7_tracking python=3.11(your puthon version)
        conda activate yolov4_tracking-gpu
        
## pip required

        # TensorFlow CPU
            pip install -r requirements.txt

        # TensorFlow GPU
            pip install -r requirements-gpu.txt
           
           
## Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
         Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository. https://developer.nvidia.com/cuda-10.1-download-archive-update2




## Downloading Official YOLOv7 Pre-trained Weights

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

download any pre-trained model from the above list



## Running the Tracker with YOLOv7
        Run the code with mentioned command below (by default, pretrained yolov7 weights will be automatically downloaded into the working directory if they don't already exist).
        
        
#for specific class (person)
python detect_or_track.py --weights yolov7.pt --source "footage.mp4" --classes 0
or
python detect_or_track.py --weights yolov7.pt --no-trace --view-img --nosave --source footage.mp4 --show-fps --seed 2 --track --classes 0
        
#for detection only
python detect.py --weights yolov7.pt --source "footage.mp4"

#if you want to change source file
python detect_or_track.py --weights yolov7.pt --source "footage.mp4"

#for WebCam
python detect_or_track.py --weights yolov7.pt --source 0

#for External Camera
python detect_or_track.py --weights yolov7.pt --source 1

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python detect_or_track.py --source "your IP Camera Stream URL" --device 0

#for colored tracks 
python detect_or_track.py --weights yolov7.pt --source "street.mp4" --colored-trk

#for saving tracks centroid, track id and bbox coordinates
python detect_or_track.py --weights yolov7.pt --no-trace --view-img --nosave --source street.mp4 --show-fps --seed 2 --track --classes 0 --show-track --unique-track-color
        
Output file will be created in the working-dir/runs/detect/obj-tracking with original filename

## Results

<table>
  <tr>
    <td>YOLOv7 Detection Only</td>
    <td>YOLOv7 Object Tracking with ID</td>
    <td>YOLOv7 Object Tracking with ID and Label </td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/125909533/235033303-0c6fadee-4a1e-46b7-a19e-fab3933871c9.png"></td>
    <td><img src="https://user-images.githubusercontent.com/125909533/235035555-e2a8c3ac-3e90-402f-b5a3-f8cbd42055d9.png"></td>
     <td><img src="https://user-images.githubusercontent.com/125909533/235037903-3d4f6418-24d0-4d35-87fc-e6cdd4b4a1f0.png"></td>
  </tr>
 </table>





  
