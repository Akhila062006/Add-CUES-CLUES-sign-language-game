import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained KNN model
model = joblib.load("knn_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting live recognition...")
print("Show YES, WATER, or HELP to the camera.")
print("Press Q to quit.")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Camera error!")
        break

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb)

    prediction = "No hand detected"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks
            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            # Convert to NumPy array
            input_data = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(input_data)[0]

    # Display prediction
    cv2.putText(
        frame,
        f"Gesture: {prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Show camera
    cv2.imshow(
        "CUES & CLUES - Live Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Live recognition stopped.")