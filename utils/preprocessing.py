# utils/preprocessing.py
import cv2
import numpy as np

def preprocess_signature(img_array, size=(220, 155)):
    """
    输入图像为灰度 numpy array，输出预处理图像（统一尺寸 + 二值化 + 反色）
    """
    if img_array is None:
        raise ValueError("图像输入为空")

    img_resized = cv2.resize(img_array, size)
    _, img_thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv = cv2.bitwise_not(img_thresh)

    return img_inv
