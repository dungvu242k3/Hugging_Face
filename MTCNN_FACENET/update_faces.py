import glob
import os

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from torchvision import transforms

IMG_PATH = "./data/test_images"
DATA_PATH = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trans(img) :
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

model = InceptionResnetV1(classify = False,pretrained = "casia-webface").to(device)

model.eval()

embeddings_list = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr, "*.jpg")):
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embed = model(trans(img).to(device).unsqueeze(0))
            embeds.append(embed)

    if len(embeds) == 0:
        continue

    user_embedding = torch.cat(embeds).mean(0, keepdim=True)  # shape [1, 512]
    embeddings_list.append(user_embedding)
    names.append(usr)

embeddings = torch.cat(embeddings_list)
names = np.array(names)

if device.type == "cpu":
    torch.save(embeddings, os.path.join(DATA_PATH, "faceslistCPU.pth"))
else:
    torch.save(embeddings, os.path.join(DATA_PATH, "faceslist.pth"))

np.save(os.path.join(DATA_PATH, "usernames.npy"), names)
print("Hoàn thành! Có {0} người trong facelist.".format(names.shape[0]))
