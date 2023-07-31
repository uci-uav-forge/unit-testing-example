from src.segmentation import segment_image

import cv2
import numpy as np
import torch
from torchmetrics import JaccardIndex

def test_runs_without_error(benchmark):
    rand_img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    pred_mask, pred_colors = benchmark(segment_image, rand_img)
    assert pred_mask.shape == rand_img.shape[:-1]
    assert pred_colors.shape == (3, 3)

def test_with_all_black_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pred_mask, pred_colors = segment_image(img)
    assert pred_mask.shape == img.shape[:-1]
    assert pred_colors.shape == (3, 3)
    # assert all predicted colors are black
    assert all([np.all(color == np.array([0, 0, 0])) for color in pred_colors])

def test_with_concentric_circles():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    SHAPE_COLOR = (255,0,0)
    LETTER_COLOR = (0,255,0)
    cv2.circle(img, (50, 50), 40, SHAPE_COLOR, -1)
    cv2.circle(img, (50, 50), 20, LETTER_COLOR, -1)
    pred_mask, pred_colors = segment_image(img)
    assert pred_mask.shape == img.shape[:-1]
    assert pred_colors.shape == (3, 3)

    background_color, shape_color, letter_color = pred_colors
    assert np.all(shape_color == SHAPE_COLOR)
    assert np.all(letter_color == LETTER_COLOR)
    assert np.all(background_color == np.array([0, 0, 0]))

def assert_iou_above_threshold(threshold, do_visualize=False):
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
        if do_visualize and iou<threshold:
            cv2.imshow("mask", mask*120)
            cv2.imshow("pred_mask", pred_mask*120)
            cv2.waitKey(0)
        ious.append(iou)
    
    print(f"IOU Threshold: {threshold}")
    print(ious)
    print(f"Mean IOU: {np.mean(ious)}")
    assert all([iou > threshold for iou in ious])

# The segmentation model need not pass all these tests. They are here to show the model performance at a glance to measure improvement or regression in performance.

def test_iou_above_threshold_25_percent():
    assert_iou_above_threshold(0.25)

def test_iou_above_threshold_50_percent():
    assert_iou_above_threshold(0.5)

def test_iou_above_threshold_75_percent():
    assert_iou_above_threshold(0.75)