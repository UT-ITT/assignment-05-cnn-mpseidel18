import cv2
import numpy as np
import tensorflow as tf
import time
from pynput.keyboard import Key, Controller, Listener
import os

model_path = './01-hyperparameters/gesture_recognition.keras' 
if not os.path.exists(model_path):
    print("Please run this script from the assignment root directory (assignment-05-cnn-mpseidel18)")
    exit()

label_names = ['like', 'rock', 'peace']
CONFIDENCE_THRESHOLD = 0.80
EDGE_DENSITY_THRESHOLD = 800.0
COOLDOWN_SKIP = 2
COOLDOWN_VOLUME = 0.5
COOLDOWN_PLAY = 1

keyboard = Controller()

# load Model
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

def preprocess_image(img):
    img_resized = cv2.resize(img, (64, 64))
    return img_resized.astype('float32') / 255.0

# init webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# roi
roi_size = 250
last_action_time = 0
current_cooldown = 0

# global quit flag
running = True

def on_press(key):
    global running
    try:
        if key.char == 'q':
            running = False
    except AttributeError:
        pass

# check for k
listener = Listener(on_press=on_press)
listener.start()

print("Press 'q' ANYWHERE on your computer to quit.\n")

while running:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # flip frame so its easier to place the hand
    frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    
    # define roi
    x1 = w - roi_size - 20
    y1 = int(h / 2 - roi_size / 2)
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    
    # draw roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place Hand Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # extract roi
    roi = frame[y1:y2, x1:x2]
    
    #use canny to detect hand

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0) # Blur to remove background noise/shadows
    edges = cv2.Canny(blurred_roi, 30, 100)
    edge_density = np.sum(edges) / 255.0
    
    current_time = time.time()
    action_triggered = False
    status_text = "No Hand Detected"
    
    # check for real edges
    if edge_density > EDGE_DENSITY_THRESHOLD:
        status_text = "Hand Detected - Analyzing..."
        
        # predict first so we can assign custom cooldowns (volume up = fast spam, skip = slow)
        roi_for_cnn = cv2.flip(roi, 1)
        preprocessed = preprocess_image(roi_for_cnn)
        pred = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)
        
        confidence = np.max(pred[0])
        pred_idx = np.argmax(pred[0])
        
        if confidence > CONFIDENCE_THRESHOLD and pred_idx < len(label_names):
            gesture = label_names[pred_idx]
            status_text = f"Gesture: {gesture} ({confidence*100:.1f}%)"
            
            # set cooldown
            if gesture == 'rock':
                req_cooldown = COOLDOWN_SKIP
            elif gesture == 'like':
                req_cooldown = COOLDOWN_VOLUME
            elif gesture == 'peace':
                req_cooldown = COOLDOWN_PLAY
            else:
                req_cooldown = 1.0

            # only trigger if cooldown is over
            if current_time - last_action_time > req_cooldown:
                # trigger action
                if gesture == 'peace':
                    print("Action: Play/Pause Track")
                    keyboard.press(Key.media_play_pause)
                    keyboard.release(Key.media_play_pause)
                elif gesture == 'like':
                    print("Action: Increase Volume")
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                elif gesture == 'rock':
                    print("Action: Skip Track")
                    keyboard.press(Key.media_next)
                    keyboard.release(Key.media_next)
                
                action_triggered = True
                current_cooldown = req_cooldown
    else:
        status_text = "Wall / No Gesture"
        
    if action_triggered:
        last_action_time = current_time
        
    # show cooldown status
    if current_time - last_action_time <= current_cooldown:
        cv2.putText(frame, f"COOLDOWN ({current_cooldown}s)", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # show status
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Edges: {edge_density:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Gesture Media Controller", frame)
    cv2.imshow("Edges Debug", edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
