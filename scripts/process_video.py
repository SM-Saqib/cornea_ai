# process_video.py

import cv2
import torch
import numpy as np
from models.yolo.yolo_detector import YoloDetector
from models.resnet.resnet_classifier import ResnetClassifier
from models.mog2.mog2_detector import Mog2Detector

def preprocess_frame(frame):
    """
    Preprocesses the video frame.

    Args:
        frame (numpy.ndarray): Input video frame.

    Returns:
        numpy.ndarray: Preprocessed frame.
    """
    return cv2.resize(frame, (416, 416))

def draw_bounding_box(frame, box, label):
    """
    Draws a bounding box with a label on the frame.

    Args:
        frame (numpy.ndarray): Input video frame.
        box (tuple): Bounding box coordinates (x, y, w, h).
        label (str): Label to display.
    """
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def process_video(video_path, yolo_weights, yolo_config, resnet_weights, output_path):
    """
    Processes the video using the vision pipeline.

    Args:
        video_path (str): Path to the input video.
        yolo_weights (str): Path to YOLO weights.
        yolo_config (str): Path to YOLO config file.
        resnet_weights (str): Path to ResNet weights.
        output_path (str): Path to save the output video.
    """
    # Initialize detectors
    yolo_detector = YoloDetector(yolo_weights, yolo_config, device='cuda')
    resnet_classifier = ResnetClassifier(resnet_weights, device='cuda')
    mog2_detector = Mog2Detector()

    # Capture video feed
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Detect motion
        motion_detected, bounding_box = mog2_detector.detect_motion(preprocessed_frame)
        if motion_detected:
            x, y, w, h = bounding_box
            larger_box = (max(0, x - 10), max(0, y - 10), min(preprocessed_frame.shape[1], w + 20), min(preprocessed_frame.shape[0], h + 20))
            cropped_frame = preprocessed_frame[larger_box[1]:larger_box[1]+larger_box[3], larger_box[0]:larger_box[0]+larger_box[2]]

            # Detect objects using YOLO
            detections = yolo_detector.detect_objects(cropped_frame)
            for box, class_id in detections:
                x, y, w, h = box
                crop = cropped_frame[y:y+h, x:x+w]

                # Optionally classify the object using ResNet
                class_label = resnet_classifier.classify_image(crop)
                draw_bounding_box(frame, (x, y, w, h), str(class_label))

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "input_video.mp4"
    yolo_weights = "yolov3.weights"
    yolo_config = "yolov3.cfg"
    resnet_weights = "resnet50_weights.pth"
    output_path = "output_video.avi"
    process_video(video_path, yolo_weights, yolo_config, resnet_weights, output_path)
