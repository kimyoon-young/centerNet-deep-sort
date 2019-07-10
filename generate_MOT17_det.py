from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import cv2
import numpy as np
import sys
CENTERNET_PATH = 'CENTERNET_PATH/deep-sort-plus-pytorch/CenterNet/src/lib/'

sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)


    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]


        person_id = 1

        seq_no = 1
        #line = []

        detector.pause = False
        for (image_name) in image_names:
            ret = detector.run(image_name)

            bbox = ret['results'][person_id]

            # change xyxy to xywh
            bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

            #output detection box


            #line = [seq_no , -1]
            for data in bbox :

                #line.append(np.concatenate(([seq_no, 1], data, [-1,-1,-1])))

                opt.det_result_file.write("%d,-1, %f, %f, %f, %f, %f, -1,-1,-1\n" % (seq_no, data[0],data[1],data[2],data[3],data[4]))

                #opt.det_result_file.write("%s\n" % line.tolist())



            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)

            seq_no +=1
        # close
        opt.det_result_file.close()

if __name__ == '__main__':

    MODEL_PATH = './CenterNet/models/ctdet_coco_dla_2x.pth'
    ARCH = 'dla_34'

    # MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
    # ARCH = 'resdcn_18'

    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))


    seq_path = '../data/2DMOT17det/test/MOT17-{0:02d}/img1/'
    det_result_path= './det_results/MOT17-{}.txt'
    start_seq = 1
    end_seq = 14

    # generate box, conf file
    # f = open('../det_results')


    for i in range(start_seq, end_seq + 1):
        opt.demo = seq_path.format(i)

        if os.path.isdir(opt.demo):
            opt.det_result_file = open(det_result_path.format(i), 'w')
            demo(opt)



