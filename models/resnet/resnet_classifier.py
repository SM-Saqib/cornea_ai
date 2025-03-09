# resnet_classifier.py

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ResnetClassifier:
    """
    Class for classifying objects using ResNet.
    """

    def __init__(self, weights_path, device='cpu'):
        """
        Initializes the ResNet classifier.

        Args:
            weights_path (str): Path to ResNet weights.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.weights_path = weights_path
        self.device = torch.device(device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.model.eval()

    def classify_image(self, image_crop):
        """
        Classifies the given image crop.

        Args:
            image_crop (numpy.ndarray): Cropped image.

        Returns:
            str: Predicted class label.
        """
        image = Image.fromarray(image_crop)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = outputs.max(1)
        return predicted.item()
