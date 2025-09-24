import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox

DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_faces(name, num_images=20, delay=0.3):
    """Ambil foto wajah dari webcam"""
    user_dir = os.path.join(DATA_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Tidak bisa membuka webcam.")
        return

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)  # mirror
        cv2.putText(frame, f"{name}: {count+1}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Faces", frame)

        # simpan foto
        save_img = cv2.resize(frame, (160, 160))
        path = os.path.join(user_dir, f"{int(time.time()*1000)}_{count}.jpg")
        cv2.imwrite(path, save_img)

        count += 1
        time.sleep(delay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Selesai", f"Selesai ambil {count} foto untuk {name}!")

def start_capture():
    """Ambil input dari GUI"""
    name = entry_name.get().strip()
    num = entry_num.get().strip()

    if not name:
        messagebox.showwarning("Input Salah", "Masukkan nama karyawan dulu.")
        return

    try:
        num = int(num) if num else 20
    except ValueError:
        messagebox.showwarning("Input Salah", "Jumlah foto harus angka.")
        return

    collect_faces(name, num_images=num)

# ========== GUI ========== #
root = tk.Tk()
root.title("Face Collector")

tk.Label(root, text="Nama Karyawan:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
entry_name = tk.Entry(root, width=30)
entry_name.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Jumlah Foto (default 20):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_num = tk.Entry(root, width=10)
entry_num.grid(row=1, column=1, padx=10, pady=5, sticky="w")

btn_start = tk.Button(root, text="Start Capture", command=start_capture, bg="green", fg="white")
btn_start.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
