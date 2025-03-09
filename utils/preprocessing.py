# preprocessing.py

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_frame(frame):
    """
    Preprocesses the input frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        numpy.ndarray: Preprocessed frame.
    """
    # Implement frame preprocessing logic here.
    return frame

def preprocess_frame_mog2(frame):
    """
    Preprocesses the input frame for MOG2.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        numpy.ndarray: Preprocessed frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def preprocess_frame_yolo(frame):
    """
    Preprocesses the input frame for YOLO.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        numpy.ndarray: Preprocessed frame.
    """
    return cv2.resize(frame, (416, 416))

def preprocess_frame_resnet(frame):
    """
    Preprocesses the input frame for ResNet.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        torch.Tensor: Preprocessed frame.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(frame)
    return transform(image).unsqueeze(0)