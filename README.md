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

# Test speed

GPU : 1080ti 11G

![Alt Text](https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/centernet_vs_yolo3.gif)

(left) CenterNet based tracker: fps 18~23 / (right) original yolov3 version[2] : fps 8~9 


# Reference
[1] https://github.com/xingyizhou/CenterNet   
[2] https://github.com/Qidian213/deep_sort_yolov3
