if __name__ == "__main__":
    from ultralytics import YOLO
    import torch
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    DATASET_YAML = BASE_DIR / "dataset/dataset.yaml"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============================
    # Carregar modelo pré-treinado (YOLOv8n ou v8s)
    # ===============================
    model = YOLO("yolov8n.pt")  # leve, rápido
    # model = YOLO("yolov8s.pt")  # mais capacidade se tiver GPU boa

    # ===============================
    # Treinamento
    # ===============================
    model.train(
        data=str(DATASET_YAML),
        epochs=100,                  # número de épocas
        batch=16,                    # batch size
        imgsz=640,                   # tamanho das imagens
        device=DEVICE,
        name="alicate_display",      # nome do experimento
        exist_ok=True,
        project=str(BASE_DIR / "runs"),
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        augment=True,                # ativa augmentation padrão do YOLO
        flipud=0.0,                  # flip vertical
        fliplr=0.5,                  # flip horizontal
        mosaic=1.0,                  # uso de mosaic
        mixup=0.2,                   # mixup
        save_period=5,
        patience=20,
        verbose=True,
        val=True                     # ativa validação a cada época
    )

    print("✅ Treinamento concluído!")
