
import torch

class Config:

    # Model config
    NUM_CLASSES = 45
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = r"app\model_utils\checkpoint_best.pth"
    # MODEL_PATH = r"app\model_utils\checkpoint_vj.pth"
    MODEL_PATH = r'app\model_utils\age_focus_model.h5'
    UPLOAD_FOLDER = r'app\static\temp'
