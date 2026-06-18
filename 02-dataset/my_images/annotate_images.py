import cv2
import os
import json
import uuid

# Change this to your image path
image_path = './02-dataset/my_images/3.jpeg' 

img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image at {image_path}. Check the path.")
    exit()

height, width, _ = img.shape

print(f"Image loaded: {width}x{height}")

# Ask for the label before drawing the box
label = input("Enter gesture label (e.g., like, rock, peace, no_gesture): ").strip()

print("Draw a box and press SPACE or ENTER. Press 'c' to cancel.")

# Let user draw the bounding box
cv2.namedWindow("Select Hand", cv2.WINDOW_NORMAL)
bbox = cv2.selectROI("Select Hand", img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

if bbox == (0, 0, 0, 0):
    print("No bounding box selected. Exiting.")
    exit()

# Calculate normalized coordinates
x, y, box_w, box_h = bbox
norm_x = x / width
norm_y = y / height
norm_w = box_w / width
norm_h = box_h / height

# Generate a new UUID for HaGRID format
img_uuid = str(uuid.uuid4())

# Build the JSON object
annotation = {
    "bboxes": [
        [norm_x, norm_y, norm_w, norm_h]
    ],
    "labels": [
        label
    ],
    "landmarks": [],
    "leading_conf": 1.0,
    "leading_hand": "right", # Assuming right hand, you can change this if needed
    "user_id": ""
}

json_path = './02-dataset/my_images/annot-marius.json'

# Load existing JSON or create an empty dict if the file is empty/new
if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
    with open(json_path, 'r') as f:
        data = json.load(f)
else:
    data = {}

# Add the new annotation under the UUID key
data[img_uuid] = annotation

# Save back to the JSON file
with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"\n>>> Successfully added annotation for '{label}' to {json_path}")

# Rename the image file to match the UUID (required by HaGRID format loaders)
dir_name = os.path.dirname(image_path)
ext = os.path.splitext(image_path)[1]
new_image_path = os.path.join(dir_name, f"{img_uuid}{ext}")
os.rename(image_path, new_image_path)
print(f">>> Renamed image to: {new_image_path}")
