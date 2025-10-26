# model_utils.py
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image
import io

# Load MobileNetV2 once
_model = None

def get_model():
    global _model
    if _model is None:
        _model = MobileNetV2(weights="imagenet")
    return _model

def preprocess_pil_image(pil_img, target_size=(224,224)):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict_top_k(pil_img, k=5):
    model = get_model()
    x = preprocess_pil_image(pil_img)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=k)[0]  # list of (class, description, score)
    # Return simplified structure
    return [{"label": d[1].replace("_", " "), "prob": float(d[2])} for d in decoded]
