import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

class CourtLineDetector():
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

    def predict(self, image):
        self.model.eval()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(img_tensor)

        kps = prediction.squeeze(0).cpu().numpy()

        ori_height, ori_width = img_rgb.shape[:2]

        kps[::2] = kps[::2] * ori_width / 224.0
        kps[1::2] = kps[1::2] * ori_height / 224.0

        return kps
