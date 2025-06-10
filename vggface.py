import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def encode_known_faces(encodings_path: Path = Path("output/encodings_facenet.pkl"), training_dir: Path = Path("training")):
    names, embeddings = [], []

    for img_path in training_dir.rglob("*.jpg"):
        label = img_path.parent.name
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(tensor).cpu().numpy().flatten()
        embedding /= np.linalg.norm(embedding)

        names.append(label)
        embeddings.append(embedding)
        print(f"Encoded {img_path.name} - Label: {label}")

    data = {"names": names, "embeddings": embeddings}
    encodings_path.parent.mkdir(parents=True, exist_ok=True)
    with encodings_path.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(embeddings)} embeddings to {encodings_path}")

def vggface_faces_in_frame(frame, known_embeddings, known_names, threshold=0.6):
    boxes, _ = mtcnn.detect(frame)
    if boxes is None:
        return frame

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten()
        emb /= np.linalg.norm(emb)

        distances = np.linalg.norm(known_embeddings - emb, axis=1)
        min_dist = np.min(distances)

        name = "Unknown"
        if min_dist < threshold:
            idx = np.argmin(distances)
            name = known_names[idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

def vggface_from_webcam_mtcnn(encodings_location: Path = Path("output/encodings_facenet.pkl"), threshold: float = 0.6):
    if not encodings_location.exists():
        print("No encodings found. Please run --train first.")
        return

    with encodings_location.open("rb") as f:
        db = pickle.load(f)
        known_embeddings = np.array(db["embeddings"])
        known_names = db["names"]

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = vggface_faces_in_frame(frame, known_embeddings, known_names, threshold)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Webcam MTCNN + Facenet (VGGFace2)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def vggface_from_webcam_yolo(encodings_location: Path = Path("output/encodings_facenet.pkl"), threshold: float = 0.6):
    if not encodings_location.exists():
        print("No encodings found. Please run --train first.")
        return

    with encodings_location.open("rb") as f:
        db = pickle.load(f)
        known_embeddings = np.array(db["embeddings"])
        known_names = db["names"]

    cap = cv2.VideoCapture(0)
    yolo_model = YOLO("best.pt")

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor).cpu().numpy().flatten()
            emb /= np.linalg.norm(emb)

            distances = np.linalg.norm(known_embeddings - emb, axis=1)
            min_dist = np.min(distances)

            name = "Unknown"
            if min_dist < threshold:
                idx = np.argmin(distances)
                name = known_names[idx]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Webcam YOLO + Facenet (VGGFace2)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--realtime_yolo", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    if args.train:
        encode_known_faces()
    if args.realtime_yolo:
        vggface_from_webcam_yolo()
    if args.realtime:
        vggface_from_webcam_mtcnn()
