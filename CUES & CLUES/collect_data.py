import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# SETTINGS
# -----------------------------
GESTURE = input("Enter gesture name (YES/WATER/HELP): ").strip().upper()
NUM_SAMPLES = int(input("Enter number of samples: "))

# Create dataset folder
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

count = 0

print("\nStarting camera...")
print("Show your hand to the camera.")
print("Press Q to quit.\n")

while cap.isOpened() and count < NUM_SAMPLES:

    success, frame = cap.read()

    if not success:
        print("Camera error!")
        break

    # Flip camera for natural view
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Store 21 landmarks
            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            # Save sample
            filename = os.path.join(
                dataset_folder,
                f"{GESTURE}_{count + 1}.csv"
            )

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks)

            count += 1

    # Display progress
    cv2.putText(
        frame,
        f"{GESTURE}: {count}/{NUM_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("CUES & CLUES - Dataset Collection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\nCompleted! {count} samples collected for {GESTURE}.")