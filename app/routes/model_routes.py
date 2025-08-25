from app.model_utils.preprocess import preprocess_pipeline
from app.model_utils.model import coral_decode

from flask import Blueprint, render_template, current_app, request, jsonify
import torch
from torchvision import transforms
import cv2
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from rembg import remove


# def preprocess_image(img_path, order=[3,4,6,9]):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     img = preprocess_pipeline(img, order=order, augment=False)

#     post_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
#     ])
#     img = post_transform(img)
#     img = img.unsqueeze(0)  

#     return img

def preprocess_image(img_path, order=[3,4,6,9]):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- Background Removal ----
    pil_img = Image.fromarray(img)
    bg_removed = remove(pil_img)   # RGBA (may have transparency)

    # Fill transparent background with white
    # if bg_removed.mode == "RGBA":
    #     white_bg = Image.new("RGB", bg_removed.size, (255, 255, 255))  # white background
    #     white_bg.paste(bg_removed, mask=bg_removed.split()[3])  # use alpha channel as mask
    #     bg_removed = white_bg  # now RGB, no alpha

    # Convert back to numpy (RGB only)
    img_no_bg = np.array(bg_removed)

    # Save background removed image (force PNG if RGBA)
    base, ext = os.path.splitext(img_path)
    bg_rm_path = f"{base}_bg_rm.png" if bg_removed.mode == "RGBA" else f"{base}_bg_rm{ext}"
    bg_removed.save(bg_rm_path)

    # ---- Preprocessing ----
    
    img_proc = preprocess_pipeline(img_no_bg, order=order, augment=False)
    # Convert to PIL for saving (cast to uint8 if float32)
    if img_proc.dtype != np.uint8:
        img_proc_to_save = (img_proc * 255).clip(0, 255).astype(np.uint8)
    else:
        img_proc_to_save = img_proc

    img_proc_pil = Image.fromarray(img_proc_to_save)
    pre_path = f"{base}_pre.png"   # save as PNG
    img_proc_pil.save(pre_path)
    # ---- Torch Transform ----
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = post_transform(img_proc)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    return img_tensor



model_bp = Blueprint("model", __name__)

@model_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily in uploads/
    filename = secure_filename(file.filename)
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        # Preprocess + inference
        img_tensor = preprocess_image(save_path).to(current_app.device)

        with torch.no_grad():
            logits = current_app.model(img_tensor)
            pred_age = coral_decode(logits)

        return jsonify({"predicted_age": int(pred_age)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        # if os.path.exists(save_path):
        #     os.remove(save_path)
        pass