import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # We use EfficientNet as a robust feature extractor
        base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        # Strip the classifier to output raw features
        self.model = torch.nn.Sequential(*(list(base.children())[:-1]))
        self.model.eval().to(self.device)
        
    def process_image(self, img_path: str) -> np.ndarray:
        # Load and preprocess
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # HWC to CHW and normalize
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            
        # Flatten and convert to numpy
        return features.cpu().numpy().flatten()
