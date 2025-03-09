# yolo_detector.py
import cv2
import torch
import numpy as np

class YoloDetector:
    """
    Class for detecting objects using YOLOv3.
    """

    def __init__(self, weights_path, config_path, device='cpu'):
        """
        Initializes the YOLO detector.

        Args:
            weights_path (str): Path to YOLO weights.
            config_path (str): Path to YOLO configuration.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.weights_path = weights_path
        self.config_path = config_path
        self.device = torch.device(device)
        self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        if device == 'cuda':
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_objects(self, image):
        """
        Detects objects in the given image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            list: List of bounding boxes and class labels.
        """
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_names = self.model.getLayerNames()
        # Handle different types of output from getUnconnectedOutLayers()
        unconnected_out_layers = self.model.getUnconnectedOutLayers()
        if isinstance(unconnected_out_layers[0], (list, tuple)):
            output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        
        outputs = self.model.forward(output_layers)
        detections = self.model.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        result = []
        if len(indices) > 0:
            for i in indices.flatten():
                result.append((boxes[i], class_ids[i]))

        return result