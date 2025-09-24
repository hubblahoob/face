# utils.py
import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mtcnn():
    return MTCNN(image_size=160, margin=0, keep_all=False, device=device)

def get_facenet():
    # pre-trained FaceNet (InceptionResnetV1 trained on VGGFace2)
    return InceptionResnetV1(pretrained='vggface2').eval().to(device)
