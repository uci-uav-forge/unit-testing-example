from segmentation import segment_image

import cv2
import numpy as np
from torchmetrics import JaccardIndex

def test_iou_above_treshold():
    THRESHOLD=0.1
    
