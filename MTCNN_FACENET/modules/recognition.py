import torch


class FaceRecognizer:
    def __init__(self, database, threshold=0.8, device="cpu"):
        self.db = database
        self.threshold = threshold
        self.device = device
        self.names, self.embeddings = self.db.get_all()

    def recognize(self, embedding):
        if len(self.embeddings) == 0:
            return "Unknown", None, None

        diff = self.embeddings - embedding.unsqueeze(0)
        dist = torch.norm(diff, dim=1)
        min_dist, idx = torch.min(dist, dim=0)

        if min_dist.item() < self.threshold:
            return self.names[idx], min_dist.item(), idx.item()
        else:
            return "Unknown", min_dist.item(), None
