from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# --- 1️⃣ Test on dataset images ---
model.predict(source="aquarium_pretrain/test/images", save=True, conf=0.3)

# --- 2️⃣ Test on a single custom image ---
# image_path = "test_image.jpg"  # put your own image here
# results = model.predict(source=image_path, save=True, conf=0.3)
# cv2.imshow("Result", cv2.imread("runs/detect/predict/test_image.jpg"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
