# mog2_detector.py

import cv2

class Mog2Detector:
    """
    Class for detecting motion using MOG2.
    """

    def __init__(self):
        """
        Initializes the MOG2 detector.
        """
        self.mog2 = cv2.createBackgroundSubtractorMOG2()

    def detect_motion(self, frame):
        """
        Detects motion in the given frame.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            tuple: (motion_detected, bounding_box)
        """
        fg_mask = self.mog2.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        bounding_box = None

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Threshold to filter out small movements
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                bounding_box = (x, y, w, h)
                break

        return motion_detected, bounding_box
