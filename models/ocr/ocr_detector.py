# ocr_detector.py

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

class OcrDetector:
    """
    Class for detecting text using OCR.
    """

    def __init__(self, model_name='microsoft/trocr-base-handwritten', device='cpu'):
        """
        Initializes the OCR detector.

        Args:
            model_name (str): Name of the pre-trained OCR model.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)

    def detect_text(self, image_crop):
        """
        Detects text in the given image crop.

        Args:
            image_crop (numpy.ndarray): Cropped image.

        Returns:
            str: Detected text.
        """
        image = Image.fromarray(image_crop).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text
