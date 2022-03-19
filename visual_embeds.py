import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg

from utils import *

'''
This file is a modified version of this notebook: https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing#scrollTo=vKwPzLa4krkP

NOTE
You need to install an earlier version of torch and detectron2 in order to use this code

$ pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
$ pip3 install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

I suggest to create a new conda environment or use a seperate notebook to use the code in this file
In case you are not using CUDA, you can refer to detectron2 github repo and pytorch website to install the proper versions
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

MIN_BOXES=10
MAX_BOXES=100

def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    # cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg


def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model


def prepare_image_inputs(cfg, img_list, model):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images, model.backbone.size_divisibility)
    
    return images, batched_inputs

def get_features(model, images):
    features = model.backbone(images.tensor.to(device))
    return features


def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(1, box_features.shape[0], box_features.shape[1]) # depends on your config and batch size
    
    return box_features, features_list


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)

    return pred_class_logits, pred_proposal_deltas


def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes


def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes


def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach().cpu()
    cls_boxes = output_boxes.tensor.detach().reshape(output_boxes.tensor.shape[0]//80,80,4).cpu()
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]

    return keep_boxes, max_conf


def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]

    return keep_boxes


def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]


def generate_visual_embeds(image_file_names, images_path, cfg_path):
    cfg = load_config_and_model_weights(cfg_path)
    model = get_model(cfg)

    bgr_images = convert_to_BGR(image_file_names, images_path)
    
    visual_embeds = []
    for image in tqdm(bgr_images):
        # Convert Image to Model Input
        image, batched_inputs = prepare_image_inputs(cfg, [image], model)

        # Get ResNet+FPN features
        features = get_features(model, image)

        # Get region proposals from RPN
        proposals = get_proposals(model, image, features)
        
        # Get Box Features for the proposals
        box_features, features_list = get_box_features(model, features, proposals)

        # Get prediction logits and boxes
        pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)

        # Get FastRCNN scores and boxes
        boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)

        # Rescale the boxes to original image size
        output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]

        # Select the Boxes using NMS
        temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
        keep_boxes, max_conf = [], []
        for keep_box, mx_conf in temp:
            keep_boxes.append(keep_box)
            max_conf.append(mx_conf)

        # Limit the total number of boxes
        keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]

        # Get the visual embedding
        visual_embed = [torch.tensor(get_visual_embeds(box_feature, keep_box)).to(device) for box_feature, keep_box in zip(box_features, keep_boxes)]
        visual_embeds.append(visual_embed[0])

        return visual_embeds