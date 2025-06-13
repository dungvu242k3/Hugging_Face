import os
from datetime import datetime

import cv2
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_PATH = "./data/test_images/"
usr_name = input("Input your name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
os.makedirs(USR_PATH, exist_ok=True)

mtcnn = MTCNN(margin=20, keep_all=True, select_largest=False, post_process=False, device=device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

while cap.isOpened():
    isSuccess, frame = cap.read()
    if not isSuccess:
        break

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Face Capturing (press S to capture, ESC to exit)', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        face_img = mtcnn(frame)
        if face_img is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = os.path.join(USR_PATH, f"{timestamp}_{count}.jpg")
            mtcnn(frame, save_path=path)
            print(f"[+] Captured image {count + 1} saved to: {path}")
            count += 1
        else:
            print("[-] No face detected. Try again.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
