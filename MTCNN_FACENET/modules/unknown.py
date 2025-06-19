import cv2
import numpy as np


class UnknownFaceHandler:
    def __init__(self, database):
        self.database = database
        self.unknown_faces = []

    def add_unknown_face(self, embedding, face_img):
        self.unknown_faces.append((embedding, face_img))

    def has_unknown_faces(self):
        return len(self.unknown_faces) > 0

    def process_unknown_faces(self):
        print("\n==> Enter names for unknown faces:")
        for idx, (embedding, face_img) in enumerate(self.unknown_faces):
            window_name = f"Unknown Face {idx + 1}"
            bgr_face = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, bgr_face)

            name = input(f"Enter name for {window_name}: ").strip()
            if name:
                self.database.add_face(name, embedding.cpu())
                print(f" Added {name} to database.")
            cv2.destroyWindow(window_name)

        self.database.save_database()
        self.clear()

    def clear(self):
        self.unknown_faces = []
