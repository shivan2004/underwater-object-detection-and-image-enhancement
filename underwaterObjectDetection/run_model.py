# from ultralytics import YOLO
# import cv2
# import os
# import matplotlib.pyplot as plt
#
# # Load model
# model = YOLO("model/best.pt")   # Update path if needed
#
# # Input image
# image_path = "test_images/fish3.jpg"
#
# # Run prediction and save output in /predict/ folder
# results = model.predict(
#     source=image_path,
#     save=True,
#     project="predict",
#     name=".",
#     exist_ok=True,
#     conf=0.4
# )
#
# # Output path
# output_image = os.path.join("predict", image_path)
#
# # Load image
# img = cv2.imread(output_image)
#
# if img is None:
#     print(f"❌ Could not load predicted image: {output_image}")
# else:
#     # Convert BGR → RGB for matplotlib
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # Display the image
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title("Detected Objects")
#     plt.show()


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

