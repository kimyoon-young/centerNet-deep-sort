import os
import cv2
import numpy as np
from imutils.video import FileVideoStream


#CenterNet
import sys
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

#input_type
opt.input_type = 'vid'   # video : vid,  webcam : webcam, ip camera : ipcam

#------------------------------
# for video
opt.vid_path = 'MOT16-11.mp4'  #
#------------------------------
# for webcam  (webcam device index is required)
opt.webcam_ind = 0
#------------------------------
# for ipcamera (camera url is required.this is dahua url format)
opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
#------------------------------


from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time


def bbox_to_xywh_cls_conf(bbox):
    person_id = 1
    # only person
    bbox = bbox[person_id]

    if any(bbox[:, 4] > opt.vis_thresh):

        bbox = bbox[bbox[:, 4] > opt.vis_thresh, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

        return bbox[:, :4], bbox[:, 4]

    else:

        return None, None


class Detector(object):
    def __init__(self, opt):
        #self.vdo = cv2.VideoCapture()


        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        # self.deepsort = DeepSort("deep/checkpoint/ori_net_last.pth")


        self.write_video = True

    def open(self):

        if opt.input_type == 'webcam':
            self.vdo = FileVideoStream(opt.webcam_ind).start()

        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()

            self.vdo = FileVideoStream(opt.ipcam_url.format(key, secret, opt.ipcam_no)).start()

        # video
        else :
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo = FileVideoStream(opt.vid_path).start()


        self.im_width = int(self.vdo.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))


        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo1.avi", fourcc, 20, (self.im_width, self.im_height))
        #return self.vdo.isOpened()



    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        #while self.vdo.grab():
        while self.vdo.more():

            frame_no +=1
            start = time.time()
            #_, ori_im = self.vdo.retrieve()
            ori_im = self.vdo.read()

            #im = ori_im[ymin:ymax, xmin:xmax]
            im = ori_im
            #im = ori_im[ymin:ymax, xmin:xmax, :]

            #start_center =  time.time()

            results = self.detector.run(im)['results']
            bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(results)

            if bbox_xywh is not None:

                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))


            end = time.time()
            #print("deep time: {}s, fps: {}".format(end - start_deep_sort, 1 / (end - start_deep_sort)))

            fps =  1 / (end - start )

            avg_fps += fps
            print("centernet time: {}s, fps: {}, avg fps : {}".format(end - start, fps,  avg_fps/frame_no))

            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vdo.stop()

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
    det.open()
    det.detect()





