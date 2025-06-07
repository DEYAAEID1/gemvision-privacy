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

    predicted_label = model.config.id2label[predicted_class_idx]

import csv
import os
import datetime

FEEDBACK_CSV = "feedback_log.csv"

@app.post("/feedback")
async def submit_feedback(feedback: str = File(...)):
    # تأكد من وجود الملف، وإذا مش موجود أضف رؤوس الأعمدة
    file_exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "feedback"])  # رؤوس الأعمدة

        writer.writerow([datetime.datetime.now().isoformat(), feedback])

    return {"message": "تم استلام ملاحظتك، شكرًا!"}

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"message": "تم استلام ملاحظتك، شكرًا!"}

