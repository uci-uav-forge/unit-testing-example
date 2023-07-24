from src.segmentation import segment_image

import os

import cv2
import numpy as np
import torch
from torchmetrics import JaccardIndex

def test_runs_without_error():
    rand_img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    pred_mask, pred_colors = segment_image(rand_img)
    assert pred_mask.shape == rand_img.shape[:-1]
    assert pred_colors.shape == (3, 4)


def test_iou_above_threshold():
    THRESHOLD=0.1
    IMAGES_FOLDER = "tests/segmentation_images"

    images_and_masks = []
    for img_idx in range(4):
        image = cv2.imread(f"{IMAGES_FOLDER}/crop{img_idx}.png")
        mask  = cv2.imread(f"{IMAGES_FOLDER}/mask{img_idx}.png", cv2.IMREAD_GRAYSCALE)

        # quantize to 3 classes (0,1,2)
        mask[mask == 255] = 2
        mask[mask > 2] = 1

        images_and_masks.append((image, mask))

    jaccard_index = JaccardIndex(task='multiclass', num_classes=3)

    ious = []
    for img, mask in images_and_masks:
        pred_mask, pred_colors = segment_image(img)
        iou = jaccard_index(torch.from_numpy(pred_mask).flatten(), torch.from_numpy(mask).flatten())
        ious.append(iou)
    
    assert all([iou > THRESHOLD for iou in ious])

test_iou_above_threshold()