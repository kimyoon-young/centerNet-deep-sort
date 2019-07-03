# centerNet + deep sort with pytorch 

This is multi-people tracking code ( [centerNet[1]](https://github.com/xingyizhou/CenterNet) version of [yolov + deepsort[2]](https://github.com/ZQPei/deep_sort_pytorch) ), which implemented on CUDA 9.0, ubuntu 16.04, and Anaconda python 3.6. We used CenterNet for real-time object tracking.

# Install


```
conda env create -f CenterNet.yml
pip install -r requirments.txt
```


# Quick Start

1. Change CENTERNET_ROOT to your local directory in demo_centernet_deepsort.py.

```
CENTERNET_PATH = 'CENTERNET_ROOT/CenterNet/src/lib/'

to

e.g) CENTERNET_PATH = '/home/kyy/centerNet-deep-sort/CenterNet/src/lib/'
```


2. Run demo 

Using sample video, we can track multi person.   

```
python demo_centernet_deepsort.py
```

In test step, we used 'ctdet_coco_dla_2x.pth' model in [centernet model zoo](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md).   
Change two lines if want to use another model(e.g resdcn18.pth).

```
#MODEL_PATH = './CenterNet/models/ctdet_coco_dla_2x.pth'
#ARCH = 'dla_34'

to

MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
ARCH = 'resdcn_18'
```


# Model Performance 
## Speed comparison (centerNet vs yolov3)

GPU : one 1080ti 11G

![Alt Text](https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/centernet_vs_yolo3.gif)

(Left) CenterNet based tracker: fps 18-23  /  (Rright) original yolov3 version[2] : fps 8-9 


Additionally, fps 30~35 for ctdet_coco_resdcn18 model
   
## Person detection evalution

[coco API](https://github.com/cocodataset/cocoapi) provides the mAP evaluation code on coco dataset. So we changed that code slightly to evaluate AP for person class (line 458-464 in 'cocoapi/PythonAPI/pycocotools/cocoeval.py' same as **'tools/cocoeval.py'**).

The result is like below.   

dataset : [coco 2017 train / val](http://cocodataset.org/#download).   
model : ctdet_coco_resdcn18 model   

```
category : 0 : 0.410733757610904 #person AP
category : 1 : 0.20226150054237374 #bird AP
....
category : 79 : 0.04993736566987926
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.280 #original
```    

**AP50 comparsion** 

| model  | (person) AP50 | (all classes) AP50 |
| ------------- | ------------- | ------------- |
| ctdet_coco_dla_2x | 77.30 | 55.13 |
| ctdet_coco_resdcn18 | 68.24 | 44.9 | 
| *yolov3 416 | 66.99 | 49.02 |  


*we train and evaluate [yolov3 model](https://drive.google.com/file/d/1PIGdBHmtUu3DKxBhqmW2gfj1ujLRzZcR/view?usp=sharing) using [coco 2017 train / val dataset](http://cocodataset.org/#download) and [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) code (iteration number : 200K , avg loss : 2.8xx, batch size: 64, subdivision : 16 // in case of 161K (2000 x 80 class) [model](https://drive.google.com/file/d/1izRyBvQ3gYiDZDtHT7PEaQMCgmAsq9XB/view?usp=sharing), AP50 is 65.02 (person) / 48.54 (all classes)). 

# Reference
[1] https://github.com/xingyizhou/CenterNet   
[2] https://github.com/ZQPei/deep_sort_pytorch   
[3] https://github.com/AlexeyAB/darknet   


