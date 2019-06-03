# centerNet + deep sort with pytorch 

This code is centerNet[1] version of yolov + deepsort[2], which implemented on CUDA 9.0, ubuntu 16.04, and python 3.6.


# Install

```
conda create -f CenterNet.yml
pip install -r requirments.txt
```

# Test results (1080ti 11G)
For my FHD video (30fps), 
fps 17~20 compared to fps 8 from original yolov3 version[2]



# Reference
[1] https://github.com/xingyizhou/CenterNet
[2] https://github.com/Qidian213/deep_sort_yolov3
