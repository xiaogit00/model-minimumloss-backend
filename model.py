import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import requests

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # ImageNet class labels
        self.labels = self.load_labels()
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_labels(self):
        """Load ImageNet class labels"""
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            response = requests.get(url)
            return response.json()
        except:
            # Fallback labels
            return [f"Class {i}" for i in range(1000)]
    
    def predict(self, image_bytes):
        """Predict image class"""
        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(5):
            idx = top5_indices[i].item()
            predictions.append({
                'class': self.labels[idx] if idx < len(self.labels) else f"Class {idx}",
                'probability': top5_prob[i].item() * 100
            })
        
        return predictions

# Singleton instance
classifier = ImageClassifier()