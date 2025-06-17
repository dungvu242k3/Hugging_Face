import cv2
import torch

from modules.camera import Camera
from modules.database import FaceDatabase
from modules.detector import FaceDetector
from modules.embeding import FaceEmbedder
from modules.recognition import FaceRecognizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FaceDetector(device)
    embedder = FaceEmbedder(device)
    database = FaceDatabase("database/database.pkl")
    recognizer = FaceRecognizer(database, device=device)
    camera = Camera()

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break
        boxes = detector.detect_faces(frame)
        faces = detector.extract_faces(frame, boxes) if boxes is not None else []


        if not faces:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        for box, face_img in zip(boxes, faces):
            embedding = embedder.get_embedding(face_img)
            name, dist, _ = recognizer.recognize(embedding)

            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name == "Unknown":
                key = cv2.waitKey(1) & 0xFF
                if key == ord("a"):
                    user_name = input("Enter name: ")
                    database.add_face(user_name, embedding.cpu())
                    database.save_database()
                    recognizer = FaceRecognizer(database, device=device)
                    print(f"Added {user_name} to database.")


        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
