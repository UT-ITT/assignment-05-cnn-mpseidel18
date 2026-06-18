import os
import cv2
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# define path
model_path = './01-hyperparameters/gesture_recognition.keras'
directories = [
    ('./02-dataset/my_images', 'annot-marius.json'),
    ('./02-dataset/tutor_images', 'annot-tutors.json')
]
output_cm_path = './02-dataset/conf-matrix.png'

# load the trained model
print("Loading model...")
model = tf.keras.models.load_model(model_path)

# label mappings (based on assignment instructions)
label_names = ['like', 'rock', 'peace'] 
# adding 'no_gesture' so it appears on the confusion matrix axis
cm_labels = ['like', 'rock', 'peace', 'no_gesture']

def preprocess_image(img):
    img_resized = cv2.resize(img, (64, 64))
    return img_resized.astype('float32') / 255.0

y_true = []
y_pred = []

print("Extracting images and making predictions...")
for dir_path, json_name in directories:
    json_path = os.path.join(dir_path, json_name)
    if not os.path.exists(json_path):
        print(f"Annotation file not found: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        annotations = json.load(f)
        
    for filename in os.listdir(dir_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        uuid = os.path.splitext(filename)[0]
        if uuid not in annotations:
            continue
            
        img_path = os.path.join(dir_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        annot = annotations[uuid]
        
        # go through each bounding box
        for i, bbox in enumerate(annot['bboxes']):
            # convert normalized coordinates back to pixel coordinates
            x1 = int(bbox[0] * img.shape[1])
            y1 = int(bbox[1] * img.shape[0])
            w = int(bbox[2] * img.shape[1])
            h = int(bbox[3] * img.shape[0])
            x2 = x1 + w
            y2 = y1 + h
            
            # crop to the bounding box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            preprocessed = preprocess_image(crop)
            
            # make prediction
            pred = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)
            pred_idx = np.argmax(pred[0])
            
            # convert index back to string label
            if pred_idx < len(label_names):
                pred_label = label_names[pred_idx]
            else:
                pred_label = 'unknown'
                
            true_label = annot['labels'][i]
            
            y_true.append(true_label)
            y_pred.append(pred_label)

print("\n--- Evaluation ---")
print("True Labels: ", y_true)
print("Pred Labels: ", y_pred)

# plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix: My Images & Tutors')
plt.xticks(rotation=45)

# save the plot
plt.tight_layout()
plt.savefig(output_cm_path)
print(f"\nSaved confusion matrix to {output_cm_path}")
