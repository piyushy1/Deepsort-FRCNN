import os
import cv2
import numpy as np

from deepsort.deep_sort import DeepSort

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

class Detector(object):
    def __init__(self):
        # self.vdo = cv2.VideoCapture()
        # self.write_video = True
        # self.class_names = self.yolo3.class_names
        # self.count = 0
        # fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
        # self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (600, 500))
        self.deepsort = DeepSort("deepsort/deep/checkpoint/ckpt.t7")

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (self.im_width,self.im_height))
        return self.vdo.isOpened()
        

    def update_deepsort(self, new_item):
        # xmin, ymin, xmax, ymax = self.area    
        # bbox_xywh, cls_conf, cls_ids = self.yolo3(im)
        # print(cls_ids)
        bbox_xyxy, cls_conf, image = new_item
        ori_im = image
        bbox_xyxy = bbox_xyxy.to('cpu').data.numpy()
        cls_conf = cls_conf.to('cpu').data.numpy()

        # for i in bbox_xyxy:
        #     cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])),(255,255,255), 2)

        bbox_xywh = []
        for i in bbox_xyxy:
            x1,y1,x2,y2 = i
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            bbox_xywh.append([x1,y1, w, h])

        # for i in bbox_xywh:
        #     cv2.rectangle(image, (int(i[0]), int(i[1])), ( int(i[0]) + int(i[2]), int(i[1]) + int(i[3])),(255,255,255), 2)

        
        bbox_xywh = np.array(bbox_xywh)
        # print(ori_im.shape)
        # im = ori_im[:, :, (2,1,0)]

        if bbox_xywh is not None:
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1]
                return {'identities': identities, 'bbox_xyxy': bbox_xyxy}
                # ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin,ymin))