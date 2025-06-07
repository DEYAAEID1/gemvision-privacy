from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import io
import torch.nn as nn
import gdown
from torchvision import transforms, models

app = FastAPI(title="GemVision – تصنيف نوع ولون الحجر الكريم")

# ========== نموذج ViT (اسم الحجر) ==========
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# ========== نموذج الألوان (مدرَّب محليًا) ==========
color_labels = ['black', 'blue', 'green', 'purple', 'red', 'yellow']

color_model = models.resnet18(weights=None)
color_model.fc = nn.Linear(color_model.fc.in_features, len(color_labels))
model_url = "https://drive.google.com/uc?id=1AbCdEfGH1234XYZ"  # استبدل بـ FILE_ID تبعك
model_path = "gem_color_classifier.pt"
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

color_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
color_model.eval()

color_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Endpoint /analyze ==========
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # ----- 1. تحليل ViT (اسم الحجر) -----
    vit_inputs = vit_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vit_outputs = vit_model(**vit_inputs)
        vit_logits = vit_outputs.logits
        stone_idx = vit_logits.argmax(-1).item()
        stone_label = vit_model.config.id2label[stone_idx]

    # ----- 2. تحليل اللون -----
    color_input = color_transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        color_logits = color_model(color_input)
        color_idx = color_logits.argmax(-1).item()
        color_label = color_labels[color_idx]

    # ----- 3. إرجاع النتيجة -----
    return JSONResponse(content={
        "stone_prediction": stone_label,
        "color_prediction": color_label,
        "note": "تم استخدام نموذجين لتحليل نوع ولون الحجر بناءً على الصورة."
    })
