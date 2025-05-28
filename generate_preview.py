import os
import cv2

# CONFIG
images_path = "datasets/dataset_pothole/images/train"
labels_path = "datasets/dataset_pothole/labels/train"
output_path = "datasets/preview/train_boxes"

# Create output folder
os.makedirs(output_path, exist_ok=True)

# Loop through all images
for image_name in os.listdir(images_path):
    if not image_name.endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(images_path, image_name)
    label_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + ".txt")

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)
                # Convert from YOLO format to pixel coordinates
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = os.path.join(output_path, image_name)
    cv2.imwrite(out_path, img)

print("✅ All labeled images saved with boxes.")
