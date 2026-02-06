import shutil
from pathlib import Path

# ===============================
# Configurações
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# Pastas a serem verificadas
SETS = ["train", "val"]

REMOVED_DIR = BASE_DIR / "removidos"
REMOVED_IMAGES = REMOVED_DIR / "images"
REMOVED_LABELS = REMOVED_DIR / "labels"

# Criar pastas de removidos se não existirem
for p in [REMOVED_IMAGES, REMOVED_LABELS]:
    p.mkdir(parents=True, exist_ok=True)

# ===============================
# Função para checar labels
# ===============================
def is_label_valid(label_path):
    """
    Verifica se todas as coordenadas estão entre 0 e 1 e não são negativas.
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        try:
            _, x, y, w, h = map(float, parts)
        except:
            return False
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            return False
    return True

# ===============================
# Função para processar um conjunto (train ou val)
# ===============================
def check_and_remove(set_name):
    IMAGES_DIR = DATASET_DIR / f"images/{set_name}"
    LABELS_DIR = DATASET_DIR / f"labels/{set_name}"

    all_labels = list(LABELS_DIR.glob("*.txt"))
    for label_path in all_labels:
        img_path = IMAGES_DIR / f"{label_path.stem}.jpg"
        
        if not is_label_valid(label_path):
            print(f"⚠️ Label inválida detectada em {set_name}: {label_path.name}")
            
            # Mover label
            if label_path.exists():
                shutil.move(str(label_path), REMOVED_LABELS / label_path.name)
            
            # Mover imagem correspondente
            if img_path.exists():
                shutil.move(str(img_path), REMOVED_IMAGES / img_path.name)

# ===============================
# Executar para train e val
# ===============================
for s in SETS:
    check_and_remove(s)

print("✅ Checagem completa. Arquivos inválidos movidos para 'removidos/'")
