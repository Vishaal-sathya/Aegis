from app.model_utils.preprocess import preprocess_pipeline
from app.model_utils.model import coral_decode

from flask import Blueprint, render_template, current_app, request, jsonify
import torch
from torchvision import transforms
import cv2
import os
from werkzeug.utils import secure_filename


def preprocess_image(img_path, order=[3,4,6,9]):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess_pipeline(img, order=order, augment=False)

    post_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])
    img = post_transform(img)
    img = img.unsqueeze(0)  

    return img


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
        if os.path.exists(save_path):
            os.remove(save_path)
