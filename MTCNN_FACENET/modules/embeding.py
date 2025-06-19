import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms


class FaceEmbedder:
    def __init__(self, device='cpu'):
        self.device = device
        # Không được dùng classify=True
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Chuẩn hóa input theo yêu cầu của FaceNet
        ])

    def get_embedding(self, face_img):
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)  # Đảm bảo là kiểu PIL Image

        face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)  # [1, 512]
        return embedding.squeeze(0)  # [512]
