from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import glob

# Load trained model
model = YOLO("model/best1.pt")

# Folder containing input images
input_folder = "test_images3"

# Run detection on all images in folder
results = model.predict(
    source=input_folder,
    save=True,
    project="predict",
    name=".",
    exist_ok=True,
    conf=0.4
)

# Show each output image
output_folder = "predict"
image_files = glob.glob(os.path.join(output_folder, "*.*"))

if len(image_files) == 0:
    print("No output images found!")
else:
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis("off")
            plt.show()
        else:
            print("Could not load:", img_path)

