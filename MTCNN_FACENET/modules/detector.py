import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:
    def __init__(self, device='cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)
        return boxes

    def extract_faces(self, frame, boxes, size=160):
        faces = []
        h, w, _ = frame.shape

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (size, size))
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            faces.append(face)

        return faces

