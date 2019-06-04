# centerNet + deep sort with pytorch 

This code is [centerNet[1]](https://github.com/xingyizhou/CenterNet) version of [yolov + deepsort[2]](https://github.com/Qidian213/deep_sort_yolov3), which implemented on CUDA 9.0, ubuntu 16.04, and python 3.6.


# Install

```
conda create -f CenterNet.yml
pip install -r requirments.txt
```


# Run

In test step, we used 'ctdet_coco_dla_2x.pth' model in [centernet model zoo](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md)   


```
python demo_centernet_deepsort.py
```

# Speed comparison (centerNet vs yolov3)

GPU : one 1080ti 11G

![Alt Text](https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/centernet_vs_yolo3.gif)

(Left) CenterNet based tracker: fps 18-23  /  (Rright) original yolov3 version[2] : fps 8-9 


# Model performance

We use evaluate person class AP using modified tools/cocoeval.py from cocoapi/PythonAPI/pycocotools/cocoeval.py 

dataset: [coco 2017 Val images](http://cocodataset.org/#download)


(person) AP : 51.1
(all) AP : 37.4

| model  | (person) AP | (all classes) mAP | fps |
| ------------- | ------------- | ------------- | ------------- |
| ctdet_coco_dla_2x.pth | 51.1 | 37.4 | 18~23 |
| ctdet_coco_resdcn18 | 41.1 | 28.0 | 30~35 |



# Reference
[1] https://github.com/xingyizhou/CenterNet   
[2] https://github.com/Qidian213/deep_sort_yolov3
