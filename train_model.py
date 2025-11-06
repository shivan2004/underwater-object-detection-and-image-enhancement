from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model on your dataset
model.train(
    data="aquarium_pretrain/data.yaml",  # path to dataset yaml
    epochs=2,                           # increase if you want better accuracy
    imgsz=640,                           # image size
    batch=8,                             # adjust based on GPU/CPU
)
