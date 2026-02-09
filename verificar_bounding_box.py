import cv2
from os import path

img = cv2.imread(path.join("dataset_full", "images", "20260206_190146_656283.jpg"))
h, w, _ = img.shape

label_path = path.join("dataset_full", "labels", "20260206_190146_656283.txt")

with open(label_path) as f:
    for line in f:
        class_id, xc, yc, bw, bh = map(float, line.split())

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

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

cv2.imshow("verificacao", img)
cv2.waitKey(0)
