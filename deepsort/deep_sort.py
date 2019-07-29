import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import cv2

class DeepSort(object):
    def __init__(self, model_path):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=True)

        max_cosine_distance = 0.2
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        print(ori_img.shape[:2])

        # generate detections
        try:
            features = self._get_features(bbox_xywh, ori_img)
        except Exception as e:
            print(e)
        detections = [Detection(bbox_xywh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                print(track.is_confirmed())
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)

        print(outputs)
        return outputs

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        # x1 = max(int(x-w/2),0)
        # x2 = min(int(x+w/2),self.width-1)
        # y1 = max(int(y-h/2),0)
        # y2 = min(int(y+h/2),self.height-1)
        x2 = x + w
        y2 = y + h
        return int(x),int(y),int(x2), int(y2)
    
    def _get_features(self, bbox_xywh, ori_img):
        import sys
        import time
        features = []
    
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            # cv2.imshow( "patch.jpg", im)
            # time.sleep(1)
            # cv2.waitKey(1)
            # print("fasd")
            # sys.exit(0)
            feature = self.extractor(im)[0]
            features.append(feature)
        
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features



if __name__ == '__main__':
    pass
