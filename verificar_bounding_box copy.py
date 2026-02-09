import cv2
from os import path

# === Carrega imagem original ===
img = cv2.imread(path.join("dataset_full", "images", "20260206_190220_095713.jpg"))
h, w, _ = img.shape

label_path = path.join("dataset_full", "labels", "20260206_190220_095713.txt")

with open(label_path) as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue

        class_id, xc, yc, bw, bh = map(float, line.split())

        # === YOLO → pixel ===
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # Segurança
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # === Desenha bounding box (debug) ===
        color = (0, 255, 0) if int(class_id) == 0 else (255, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"class {int(class_id)}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        # === CROP ===
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # === Pré-processamento para OCR ===
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # (opcional, mas altamente recomendado)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # === Adaptive Threshold ===
        th = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # === Visualização ===
        cv2.imshow(f"crop_{idx}_original", crop)
        cv2.imshow(f"crop_{idx}_threshold", th)

# === Mostra imagem com boxes ===
cv2.imshow("verificacao_bbox", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
