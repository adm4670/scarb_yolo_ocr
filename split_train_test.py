import shutil
import random
from pathlib import Path
import yaml

# ===============================
# Configurações
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATASET_FULL = BASE_DIR / "dataset_full"      # dataset_full/images e dataset_full/labels
DATASET_DIR = BASE_DIR / "dataset"            # destino final com train/val
TRAIN_RATIO = 0.8

# Pastas do dataset YOLO
IMAGES_TRAIN = DATASET_DIR / "images/train"
IMAGES_VAL = DATASET_DIR / "images/val"
LABELS_TRAIN = DATASET_DIR / "labels/train"
LABELS_VAL = DATASET_DIR / "labels/val"

# Classe única
CLASSES = ["display", "button"]

# Cria pastas se não existirem
for p in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    p.mkdir(parents=True, exist_ok=True)

# ===============================
# Listar imagens e labels
# ===============================
images_dir = DATASET_FULL / "images"
labels_dir = DATASET_FULL / "labels"

all_images = list(images_dir.glob("*.jpg"))
print(f"🔹 Imagens encontradas: {len(all_images)}")
if not all_images:
    raise ValueError(f"Nenhuma imagem encontrada em {images_dir}")

all_labels = {img.stem: labels_dir / f"{img.stem}.txt" for img in all_images}

# ===============================
# Embaralhar e dividir
# ===============================
random.shuffle(all_images)
split_idx = int(len(all_images) * TRAIN_RATIO)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"🔹 Train: {len(train_images)}, Val: {len(val_images)}")

# ===============================
# Função para copiar imagens e labels
# ===============================
def copy_files(images_list, images_dest, labels_dest):
    for img_path in images_list:
        # Copia imagem
        shutil.copy2(img_path, images_dest / img_path.name)

        # Copia label
        label_path = all_labels.get(img_path.stem)
        if label_path.exists():
            shutil.copy2(label_path, labels_dest / label_path.name)
        else:
            print(f"⚠️ Label não encontrada para {img_path.name}")

# ===============================
# Copiar arquivos
# ===============================
copy_files(train_images, IMAGES_TRAIN, LABELS_TRAIN)
copy_files(val_images, IMAGES_VAL, LABELS_VAL)

print(f"✅ Dataset pronto em {DATASET_DIR}")

# ===============================
# Criar dataset.yaml
# ===============================
yaml_path = DATASET_DIR / "dataset.yaml"

dataset_yaml = {
    "train": str(IMAGES_TRAIN.resolve()),
    "val": str(IMAGES_VAL.resolve()),
    "nc": len(CLASSES),
    "names": CLASSES
}

with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print(f"✅ Arquivo YAML criado: {yaml_path}")
