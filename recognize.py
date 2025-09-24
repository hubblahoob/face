import cv2
import numpy as np
import os
import torch
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from utils import get_mtcnn, get_facenet, device

# Pop up
import tkinter as tk
from tkinter import messagebox

EMB_FILE = "embeddings/embeddings.npz"
ATT_FILE = "attendance.csv"
THRESHOLD = 0.6  # sesuaikan setelah uji coba

def load_embeddings():
    data = np.load(EMB_FILE, allow_pickle=True)
    return data['embeddings'], data['labels']

def build_classifier(embs, labels):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(embs, labels)
    return knn

def ensure_attendance_file():
    if not os.path.exists(ATT_FILE):
        df = pd.DataFrame(columns=['name', 'date', 'time', 'timestamp'])
        df.to_csv(ATT_FILE, index=False)

def mark_attendance(name):
    """Catat kehadiran sekali per hari"""
    ensure_attendance_file()
    df = pd.read_csv(ATT_FILE)

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    now_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    already = ((df['name'] == name) & (df['date'] == today)).any()
    if not already:
        new = pd.DataFrame([{
            'name': name,
            'date': today,
            'time': now_time,
            'timestamp': now_stamp
        }])
        df = pd.concat([df, new], ignore_index=True)
        df.to_csv(ATT_FILE, index=False)
        print(f"[{now_stamp}] Attendance recorded: {name}")
        return True
    else:
        print(f"[{now_stamp}] {name} sudah tercatat hari ini.")
        return False

def show_popup(name, recorded=True):
    """Pop up saja (tanpa suara)"""
    root = tk.Tk()
    root.withdraw()
    if recorded:
        messagebox.showinfo("Absen Berhasil", f"Absen berhasil untuk {name}")
    else:
        messagebox.showinfo("Info", f"{name} sudah absen hari ini")
    root.destroy()

def main():
    embeddings, labels = load_embeddings()
    build_classifier(embeddings, labels)
    mtcnn = get_mtcnn()
    facenet = get_facenet()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Tidak bisa membuka webcam.")
        return

    print("Starting recognition. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_tensor = None
        try:
            face_tensor = mtcnn(img)
        except Exception:
            pass

        name_to_show = "No face"
        if face_tensor is not None:
            with torch.no_grad():
                emb = facenet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]

            dist = np.linalg.norm(embeddings - emb, axis=1)
            idx = np.argmin(dist)
            min_dist = dist[idx]
            pred_label = labels[idx]

            if min_dist < THRESHOLD:
                name_to_show = f"{pred_label} ({min_dist:.3f})"
                recorded = mark_attendance(pred_label)
                cap.release()
                cv2.destroyAllWindows()
                show_popup(pred_label, recorded)
                return
            else:
                name_to_show = f"Unknown ({min_dist:.3f})"

        cv2.putText(frame, name_to_show, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
