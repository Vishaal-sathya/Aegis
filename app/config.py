
import torch
import os
    

class Config:

    # Model config
    NUM_CLASSES = 45
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # MODEL_PATH = r"app\model_utils\checkpoint_best.pth"
    # MODEL_PATH = r"app\model_utils\checkpoint_vj.pth"
    # MODEL_PATH = r'app\model_utils\age_focus_model.h5'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model_utils", "checkpoint_best.pth")
    UPLOAD_FOLDER = r'app\static\temp'
