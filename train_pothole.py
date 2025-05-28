import comet_ml
from ultralytics import YOLO


# Load a model
model = YOLO("yolo11n.pt")

# Train the model with MPS
results = model.train(data="pothole.yaml", project="pothole_detection", epochs=200, imgsz=640, max_det=100, conf=0.25 , device="mps", amp=True,
    augment=True, batch=-1,patience=20)

