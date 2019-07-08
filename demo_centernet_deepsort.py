import os
import cv2
import numpy as np

import sys

#Change path your local directory
CENTERNET_PATH = 'CENTERNET_ROOT/CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts


MODEL_PATH = './CenterNet/models/ctdet_coco_dla_2x.pth'
ARCH = 'dla_34'

#MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
#ARCH = 'resdcn_18'




TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time


class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()
      

        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
     
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo1.avi", fourcc, 20, (self.im_width, self.im_height))
        return self.vdo.isOpened()

    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        while self.vdo.grab():

            frame_no +=1
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax]


            #start_center =  time.time()

            person_id = 1
            confidence = 0.5
            # only person ( id == 1)
            bbox = self.detector.run(im)['results'][person_id]
            #bbox = ret['results'][person_id]
            bbox = bbox[bbox[:, 4] >  confidence, :]
            #box_info = ret['results']

            bbox[:, 2] =  bbox[:, 2] - bbox[:, 0] #+  (bbox[:, 2] - bbox[:, 0]) /2
            bbox[:, 3] =  bbox[:, 3] - bbox[:, 1] #+  (bbox[:, 3] - bbox[:, 1]) /2


            #start_deep_sort = time.time()


            cls_conf = bbox[:, 4]


            outputs = self.deepsort.update(bbox[:,:4], cls_conf, im)



            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))


            end = time.time()
            #print("deep time: {}s, fps: {}".format(end - start_deep_sort, 1 / (end - start_deep_sort)))

            print("centernet time: {}s, fps: {}".format(end - start, 1 / (end - start)))
            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)


if __name__ == "__main__":
    import sys

    # if len(sys.argv) == 1:
    #     print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    # else:
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 800, 600)

    #opt = opts().init()
    det = Detector(opt)

    # det.open("D:\CODE\matlab sample code/season 1 episode 4 part 5-6.mp4")
    det.open("MOT16-11.mp4")
    det.detect()
