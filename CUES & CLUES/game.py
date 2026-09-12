import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# CUES & CLUES - Sign Language Game
# ==========================================

# Load KNN model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_model.pkl"

model = joblib.load(MODEL_PATH)

# ==========================================
# MEDIAPIPE
# ==========================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# NOTE:
# The newer MediaPipe version uses the Tasks API.
# A hand-landmarker model file is required.

HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing MediaPipe model: {HAND_MODEL_PATH}"
    )

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)

# ==========================================
# GAME SETTINGS
# ==========================================

missions = [
    ("Mission 1", "You are thirsty. Show WATER.", "WATER"),
    ("Mission 2", "You need assistance. Show HELP.", "HELP"),
    ("Mission 3", "Answer the question. Show YES.", "YES")
]

score = 0
lives = 3

# ==========================================
# START SCREEN
# ==========================================

print("\n================================")
print("        CUES & CLUES")
print("================================")
print("     SIGN LANGUAGE GAME")
print("================================")
print()
print("Lives : 3")
print("Score : 0")
print()
print("Missions : 3")
print()
print("Press ENTER to start the game.")
print("Press Q to quit.")
print("================================")

choice = input()

if choice.lower() == "q":
    print("Game closed.")
    exit()

# ==========================================
# GAME START
# ==========================================

print("\nGAME STARTED!")
print("Good luck!")

# ==========================================
# MISSION LOOP
# ==========================================

for mission, instruction, correct_gesture in missions:

    if lives <= 0:
        break

    print("\n--------------------------------")
    print(mission)
    print(instruction)
    print("--------------------------------")

    cap = cv2.VideoCapture(0)

    completed = False
    correct_frames = 0
    wrong_frames = 0
    prediction = "No hand detected"

    while cap.isOpened() and not completed:

        success, frame = cap.read()

        if not success:
            print("Camera error!")
            break

        frame = cv2.flip(frame, 1)

        # ==================================
        # MEDIAPIPE HAND DETECTION
        # ==================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        detection_result = hand_landmarker.detect(
            mp_image
        )

        prediction = "No hand detected"

        # ==================================
        # HAND DETECTED
        # ==================================

        if detection_result.hand_landmarks:

            hand_landmarks = detection_result.hand_landmarks[0]

            # Draw landmarks
            height, width, _ = frame.shape

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # ==================================
            # DRAW CONNECTIONS
            # ==================================

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start, end in connections:

                x1 = int(
                    hand_landmarks[start].x * width
                )
                y1 = int(
                    hand_landmarks[start].y * height
                )

                x2 = int(
                    hand_landmarks[end].x * width
                )
                y2 = int(
                    hand_landmarks[end].y * height
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # ==================================
            # COLLECT LANDMARKS
            # ==================================

            landmarks = []

            for landmark in hand_landmarks:

                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            input_data = np.array(
                landmarks
            ).reshape(1, -1)

            # ==================================
            # PREDICT GESTURE
            # ==================================

            prediction = model.predict(
                input_data
            )[0]

            # ==================================
            # CORRECT GESTURE
            # ==================================

            if prediction == correct_gesture:

                correct_frames += 1
                wrong_frames = 0

            else:

                correct_frames = 0
                wrong_frames += 1

            # ==================================
            # MISSION COMPLETED
            # ==================================

            if correct_frames >= 10:

                score += 10
                completed = True

            # ==================================
            # WRONG GESTURE
            # ==================================

            if wrong_frames >= 30:

                lives -= 1

                wrong_frames = 0
                correct_frames = 0

                print("Wrong gesture!")
                print(
                    f"Lives remaining: {lives}"
                )

                if lives <= 0:
                    completed = True

        # ==========================================
        # DISPLAY
        # ==========================================

        cv2.putText(
            frame,
            "CUES & CLUES",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            mission,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Required: {correct_gesture}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Detected: {prediction}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Score: {score}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Lives: {lives}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Progress: {correct_frames}/10",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ==========================================
        # FEEDBACK
        # ==========================================

        if prediction == correct_gesture:

            cv2.putText(
                frame,
                "CORRECT! HOLD IT!",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        elif prediction == "No hand detected":

            cv2.putText(
                frame,
                "SHOW YOUR HAND",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "WRONG GESTURE! TRY AGAIN!",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ==========================================
        # CAMERA
        # ==========================================

        cv2.imshow(
            "CUES & CLUES - Mission",
            frame
        )

        # Q = QUIT
        if cv2.waitKey(1) & 0xFF == ord("q"):

            lives = 0
            completed = True
            break

    # ==========================================
    # CLOSE CAMERA
    # ==========================================

    cap.release()
    cv2.destroyAllWindows()

    # ==========================================
    # MISSION RESULT
    # ==========================================

    if correct_frames >= 10:

        print("Mission completed!")
        print(f"Score: {score}")
        print(f"Lives: {lives}")

    elif lives <= 0:

        print("GAME OVER!")

    else:

        print("Mission failed.")

# ==========================================
# FINAL RESULT
# ==========================================

print("\n================================")

if lives > 0:

    print("       GAME COMPLETED!")

else:

    print("       GAME OVER!")

print("================================")
print(f"Final Score: {score}")
print(f"Lives Remaining: {lives}")
print("================================")

hand_landmarker.close()
