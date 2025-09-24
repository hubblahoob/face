# enroll.py
import os
import numpy as np
from utils import get_mtcnn, get_facenet, device
from PIL import Image
import torch
import argparse

DATA_DIR = "dataset"
EMB_DIR = "embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

def create_embeddings():
    mtcnn = get_mtcnn()
    facenet = get_facenet()
    labels = []
    embs = []
    # Loop tiap folder user
    for person in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir): continue
        imgs = [os.path.join(person_dir,f) for f in os.listdir(person_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        person_embs = []
        for img_path in imgs:
            img = Image.open(img_path).convert('RGB')
            # deteksi & crop
            face = mtcnn(img)
            if face is None:
                print("No face detected for:", img_path)
                continue
            # face is tensor [3,160,160]
            with torch.no_grad():
                embedding = facenet(face.unsqueeze(0).to(device))  # (1,512 or 128) -> facenet returns 512 or 128 depending on model
            emb_np = embedding.cpu().numpy()[0]
            person_embs.append(emb_np)
        if len(person_embs) == 0:
            print(f"WARNING: No valid faces for {person}, skipping.")
            continue
        # average embedding untuk label
        avg_emb = np.mean(person_embs, axis=0)
        embs.append(avg_emb)
        labels.append(person)
        print(f"Created embedding for {person} (from {len(person_embs)} images).")
    # simpan single file npz
    out_path = os.path.join(EMB_DIR, "embeddings.npz")
    np.savez(out_path, embeddings=np.array(embs), labels=np.array(labels))
    print("Saved embeddings to", out_path)

if __name__ == "__main__":
    create_embeddings()
