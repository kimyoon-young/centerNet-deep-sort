# centerNet + deep sort with pytorch 

This code is [centerNet[1]](https://github.com/xingyizhou/CenterNet) version of [yolov + deepsort[2]](https://github.com/ZQPei/deep_sort_pytorch), which implemented on CUDA 9.0, ubuntu 16.04, and Anaconda python 3.6.


# Install


```
conda env create -f CenterNet.yml
pip install -r requirments.txt
```


# Quick Start

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
   
## person detection evalution

[coco API](https://github.com/cocodataset/cocoapi) provides the mAP evaluation code on coco dataset. So we changed that code slightly to evaluate AP for person class (line 458-464 in 'cocoapi/PythonAPI/pycocotools/cocoeval.py' same as **'tools/cocoeval.py'**).


The result is like below.   

dataset : [coco 2017 Val images](http://cocodataset.org/#download).   
model : ctdet_coco_resdcn18 model   

```
category : 0 : 0.410733757610904 #person AP
category : 1 : 0.20226150054237374 #bird AP
....
category : 79 : 0.04993736566987926
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.280 #original
```    


| model  | (person) AP | (all classes) mAP |
| ------------- | ------------- | ------------- |
| ctdet_coco_dla_2x | 51.1 | 37.4 |
| ctdet_coco_resdcn18 | 41.1 | 28.0 | 



# Reference
[1] https://github.com/xingyizhou/CenterNet   
[2] https://github.com/Qidian213/deep_sort_yolov3
