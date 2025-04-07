from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train3/weights/best.pt")  # Load trained model

image_path = "new_images/"
output_labels = "auto_labels/"  # Directory to save labels

import os
os.makedirs(output_labels, exist_ok=True)

for img_name in os.listdir(image_path):
    img_path = os.path.join(image_path, img_name)
    results = model(img_path)

    # Save auto-generated labels
    with open(os.path.join(output_labels, img_name.replace(".jpg", ".txt")), "w") as f:
        for r in results[0].boxes.data.tolist():
            x_center, y_center, width, height, conf, cls = r  # Get bbox details
            if conf > 0.3:  # Confidence threshold to avoid noise
                f.write(f"{int(cls)} {x_center} {y_center} {width} {height}\n")
