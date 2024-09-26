import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categoricalqQq

# Load the trained model
model = load_model('cnn_model.h5')

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Load the class labels
# This should match the class labels used during training
# For dynamic handling, you could load from a file or similar
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

# Create named windows
cv2.namedWindow('Detected Gesture', cv2.WINDOW_NORMAL)
cv2.namedWindow('Text Output', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    text_output_frame = np.zeros((300, 500, 3), dtype=np.uint8)  # Black background for text output

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.extend([x, y])

            if len(data_aux) < 84:
                data_aux += [0] * (84 - len(data_aux))  # Pad with zeros if less than 84 features

            prediction = model.predict(np.array([data_aux[:84]]))
            detected_class = np.argmax(prediction)
            detected_gesture = labels_dict.get(detected_class, "Unknown")

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Display gesture on the frame
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            cv2.putText(frame, f"Sign: {detected_gesture}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display gesture on the text output frame
            cv2.putText(text_output_frame, f"{detected_gesture}", (10, 50 * (results.multi_hand_landmarks.index(hand_landmarks) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(text_output_frame, "No hands detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Detected Gesture', frame)
    cv2.imshow('Text Output', text_output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
