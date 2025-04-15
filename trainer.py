from ultralytics import YOLO
import os

print(os.getcwd())
model = YOLO("/home/anshm/Documents/Realsense/yolov8m.pt")  # Load pre-trained model
model.train(data="datasets/data.yaml", epochs=100, imgsz=640)

# model = YOLO("runs/detect/train/weights/best.pt")  # Load trained model
# results = model("test_image.jpg", save=True)  # Test on an image
