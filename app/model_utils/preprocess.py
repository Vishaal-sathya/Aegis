# preprocess.py
import cv2
import numpy as np
from mtcnn import MTCNN
import albumentations as A
import os, time

# ------------------- FACE DETECTOR -------------------
DETECTOR = MTCNN()

def detect_and_crop_face(image, target_size=(224, 224), padding=0.3):
    """
    padding: fraction of the face box to expand on each side (e.g., 0.2 = 20%)
    """
    global DETECTOR
    results = DETECTOR.detect_faces(image)
    if len(results) == 0:
        print("[WARN] No face detected, resizing full image...")
        return cv2.resize(image, target_size)

    # Take largest face
    results = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)

    # --- Add padding ---
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)

    face = image[y1:y2, x1:x2]

    return cv2.resize(face, target_size)



def align_face(image, target_size=(224, 224)):
    global DETECTOR
    results = DETECTOR.detect_faces(image)
    if len(results) == 0:
        print("[WARN] No face detected for alignment, resizing full image...")
        return cv2.resize(image, target_size)

    keypoints = results[0]['keypoints']
    left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']

    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    aligned = cv2.warpAffine(image, M, (w, h))

    return detect_and_crop_face(aligned, target_size)


# ------------------- PREPROCESSING STEPS -------------------
def resize_only(image, size=(224, 224)):
    return cv2.resize(image, size)

def normalize(image):
    return image.astype(np.float32) / 255.0

def yuv_hist_equalization(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def lab_normalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    L = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    A = (A - np.mean(A)) / (np.std(A) + 1e-6)
    B = (B - np.mean(B)) / (np.std(B) + 1e-6)
    lab = np.clip(cv2.merge([L, A, B]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_CLAHE(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def high_freq_emphasis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    high_freq = cv2.Laplacian(gray, cv2.CV_64F)
    high_freq = cv2.convertScaleAbs(high_freq)
    emphasized = cv2.addWeighted(gray, 0.7, high_freq, 0.3, 0)
    emphasized = cv2.equalizeHist(emphasized)
    return cv2.cvtColor(emphasized, cv2.COLOR_GRAY2RGB)


# ------------------- AUGMENTATION -------------------
augmentation_seq = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(0, 0.02*255), p=0.5),
])

def augment_image(image):
    return augmentation_seq(image=image)["image"]


# ------------------- PIPELINE CONFIG -------------------
PIPELINE_FUNCS = {
    1: detect_and_crop_face,
    2: align_face,
    3: resize_only,
    4: yuv_hist_equalization,
    5: lab_normalization,
    6: apply_CLAHE,
    7: high_freq_emphasis,
    8: augment_image,
    9: normalize,
}

DEFAULT_ORDER = [1, 2, 3, 5, 7, 9]   # detect → align → resize → lab_norm → high_freq → normalize


def preprocess_pipeline(image, order=None, augment=False):
    if order is None:
        order = DEFAULT_ORDER.copy()

    for step in order:
        if step == 8:  # augmentation handled separately
            continue
        func = PIPELINE_FUNCS.get(step)
        if func is not None:
            image = func(image)

    if augment:
        image = PIPELINE_FUNCS[8](image)

    return image


# ------------------- BENCHMARK -------------------
def benchmark_pipeline(image_dir, num_images=50, augment=False, order=None):
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:num_images]

    times = []
    for fpath in image_files:
        image = cv2.imread(fpath)
        if image is None:
            continue

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start = time.time()
        _ = preprocess_pipeline(image, order, augment)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    total_time = np.sum(times)

    print(f"Processed {len(times)} images")
    print(f"Average time per image: {avg_time:.4f} seconds")
    print(f"Total time: {total_time:.2f} seconds")

    return avg_time, total_time
