import argparse
import os
import random
import torch

from PIL import Image,ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config


Config.setup(image_min_side=800, image_max_side=1333,
                     anchor_ratios=None, anchor_sizes="[64, 128, 256, 512]", pooler_mode=Config.POOLER_MODE,
                     rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=1000)

dataset_name = 'coco2017'
backbone_name = 'resnet50'
path_to_checkpoint = './efrcnn/model-360000.pth'
prob_thresh = 0.8
dataset_class = DatasetBase.from_name(dataset_name)
backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
              anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
              rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
model.load(path_to_checkpoint)


import time
import cv2

def _infer(ori_batch):
    batch_new_res = []
    with torch.no_grad():
        new_batch = []
        for image in ori_batch:
            pil_im = Image.fromarray(image.astype('uint8'), 'RGB')
            image_tensor, scale = dataset_class.preprocess(pil_im, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
            new_batch.append(image_tensor.unsqueeze(dim=0))
        batch = torch.cat((new_batch), dim=0)
        batch_result = model.eval().forward(batch.cuda())

        for i,image in zip(batch_result, ori_batch):
            detection_bboxes, detection_classes, detection_probs, _ = i
            detection_bboxes /= scale
            # cv2.imshow("af2", image)
            # cv2.waitKey(0)
            kept_indices = detection_probs > prob_thresh
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]

            mask = detection_classes == 3 ## 3 for car class
            # detection_classes = detection_classes[mask]
            detection_bboxes = detection_bboxes[mask]
            detection_probs = detection_probs[mask]

            # draw = ImageDraw.Draw(pil_im)

            # for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            #     color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            #     bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            #     category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

            #     draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            #     draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

            # pil_im.save("testingnow.jpg")
            # break
            batch_new_res.append((detection_bboxes, detection_probs, image))
    return batch_new_res