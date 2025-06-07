from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import io

app = FastAPI(title="GemVision Image Classifier")

# تحميل النموذج والمعالج
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    predicted_label = model.config.id2label

import datetime
import json
import os

FEEDBACK_FILE = "feedback_log.json"

@app.post("/feedback")
async def submit_feedback(feedback: str = File(...)):
    feedback_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "feedback": feedback
    }

    # إذا الملف موجود، اقرأه
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # أضف الملاحظة الجديدة
    data.append(feedback_entry)

    # احفظ التحديث
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"message": "تم استلام ملاحظتك، شكرًا!"}

