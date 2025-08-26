import torch
import os
from app.model_utils.model import AgePredictionCORAL  # import your class definition
# from tensorflow.keras.models import load_model

def load_model_pt(checkpoint_path, device="cpu", num_classes=45):
    """Load and return the trained PyTorch model."""
    model = AgePredictionCORAL(num_classes=num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle checkpoints with or without "model_state_dict"
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# def load_h5(path,compile=False):
#     model = load_model(path, compile=False)
#     return model