import os
import pickle

import torch


class FaceDatabase:
    def __init__(self, db_path="database/database.pkl"):
        self.db_path = db_path
        self.db = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def add_face(self, name, embedding):
        self.db[name] = embedding.cpu()
        self.save_database()

    def get_all(self):
        names = list(self.db.keys())
        embeddings = torch.stack([self.db[n] for n in names])
        return names, embeddings
