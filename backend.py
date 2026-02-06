from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from pydantic import BaseModel
import base64
import io
from PIL import Image, ImageDraw
from pathlib import Path
from datetime import datetime
import json

import torch
from ultralytics import YOLO
import numpy as np

# ===============================
# App
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restrinja ao domínio correto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# HTML
# ===============================
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
CAPTURE_FILE = BASE_DIR / "captura.html"
ROTULAGEM_FILE = BASE_DIR / "rotulagem.html"

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return INDEX_FILE.read_text(encoding="utf-8")

@app.get("/captura", response_class=HTMLResponse)
def serve_capture():
    return CAPTURE_FILE.read_text(encoding="utf-8")

@app.get("/rotulagem", response_class=HTMLResponse)
def serve_capture():
    return ROTULAGEM_FILE.read_text(encoding="utf-8")

# ===============================
# YOLO – GPU
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Carregando YOLO ({DEVICE.upper()})...")
# model = YOLO("yolov8n.pt")
model = YOLO("best.pt")
model.to(DEVICE)
print("✅ YOLO pronto")

# ===============================
# API Payload
# ===============================
class FramePayload(BaseModel):
    image: str

class LabelPayload(BaseModel):
    filename: str              # Nome da imagem temporária
    labels: list               # Lista de dicts: {class_id, x, y, w, h} normalizados

# ===============================
# Endpoint /frame – YOLO inference
# ===============================
@app.post("/frame")
def receive_frame(payload: FramePayload):
    header, encoded = payload.image.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = image.copy()
    h, w = frame.height, frame.width

    # ===============================
    # Inferência YOLO
    # ===============================
    results = model.predict(
        source=np.array(frame),
        imgsz=640,
        conf=0.4,
        device=0 if DEVICE == "cuda" else "cpu",
        verbose=False
    )

    draw = ImageDraw.Draw(frame)
    detections_count = 0

    for r in results:
        if r.boxes is None:
            continue
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = model.names[int(cls_id)]
            confidence = float(conf)
            label = f"{class_name} {confidence:.2f}"

            # ===============================
            # PRINT do nome da label
            # ===============================
            print(f"🎯 Detectado: {class_name} com confiança {confidence:.2f}")

            # ===============================
            # Desenho no frame
            # ===============================
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            text_size = draw.textbbox((0, 0), label)
            draw.rectangle([(x1, y1 - text_size[3] - 6), (x1 + text_size[2] + 6, y1)], fill="red")
            draw.text((x1 + 3, y1 - text_size[3] - 3), label, fill="white")

            detections_count += 1

    if detections_count:
        print(f"✅ Total de detecções: {detections_count}")

    # Encode de volta para base64
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"status": "ok", "image": f"data:image/jpeg;base64,{encoded_image}"}


# ===============================
# Endpoint /capture – salvar imagem para rotulagem
# ===============================
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_DIR.mkdir(exist_ok=True)

@app.post("/capture")
def capture_image(payload: FramePayload):
    header, encoded = payload.image.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = CAPTURE_DIR / f"{timestamp}.jpg"
    image.save(filename, format="JPEG", quality=90)

    print(f"📸 Imagem capturada e salva: {filename.name}")
    return {"status": "ok", "filename": filename.name}

# ===============================
# Endpoint /label – salvar rotulagem YOLO
# ===============================
DATASET_DIR = BASE_DIR / "dataset_full"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/next_label")
def next_label():
    """
    Retorna a próxima imagem para rotulagem.
    """
    # Lista as imagens temporárias
    images = sorted(CAPTURE_DIR.glob("*.jpg"))

    if not images:
        return {"status": "empty", "message": "Não há imagens para rotular"}

    next_image = images[0]  # pega a primeira da lista

    # Abre e converte para base64
    with open(next_image, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "status": "ok",
        "filename": next_image.name,
        "image": f"data:image/jpeg;base64,{encoded_image}"
    }

@app.post("/label")
def label_image(payload: LabelPayload):
    """
    Recebe a imagem temporária e os labels do front, salva no formato YOLO e remove a imagem temporária.
    """
    tmp_file = CAPTURE_DIR / payload.filename
    if not tmp_file.exists():
        return {"status": "error", "message": "Imagem não encontrada"}

    # Abrir imagem
    image = Image.open(tmp_file)
    w, h = image.width, image.height

    # Salvar imagem no dataset
    dst_image_path = IMAGES_DIR / payload.filename
    image.save(dst_image_path, format="JPEG", quality=90)

    # Criar arquivo YOLO .txt
    txt_path = LABELS_DIR / (payload.filename.replace(".jpg", ".txt"))
    with open(txt_path, "w") as f:
        for label in payload.labels:
            # class_id, x, y, w, h (normalizados)
            f.write(f"{label['class_id']} {label['x']} {label['y']} {label['w']} {label['h']}\n")

    # Remover imagem temporária
    tmp_file.unlink()
    print(f"✅ Imagem rotulada e movida para dataset: {payload.filename}")

    return {"status": "ok", "image": str(dst_image_path.name), "labels": len(payload.labels)}



# ===============================
# Endpoint /delete – mover imagem para delete/
# ===============================
DELETE_DIR = BASE_DIR / "delete"
DELETE_DIR.mkdir(exist_ok=True)

class DeletePayload(BaseModel):
    filename: str

@app.post("/delete")
def delete_image(payload: DeletePayload):
    tmp_file = CAPTURE_DIR / payload.filename
    if not tmp_file.exists():
        return {"status": "error", "message": "Imagem não encontrada"}

    dst_file = DELETE_DIR / payload.filename
    tmp_file.rename(dst_file)  # move para delete/

    print(f"🗑 Imagem movida para delete/: {payload.filename}")
    return {"status": "ok", "message": f"{payload.filename} movida para delete/"}
