import cv2
import mediapipe as mp
import numpy as np
import joblib

# ==========================================
# CUES & CLUES - Sign Language Game
# ==========================================

model = joblib.load("knn_model.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
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
print("        CUES & CLUES 🎮")
print("================================")
print("     SIGN LANGUAGE GAME")
print("================================")
print()
print("❤️ Lives : 3")
print("⭐ Score : 0")
print()
print("Missions : 3")
print()
print("Press ENTER to start the game.")
print("Press Q to quit.")
print("================================")

choice = input()

if choice.lower() == "q":
    print("Game closed.")
    hands.close()
    exit()

# ==========================================
# GAME START
# ==========================================

print("\n🎮 GAME STARTED!")
print("Good luck! 🤟")

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction = "No hand detected"

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Collect landmarks
                landmarks = []

                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)

                input_data = np.array(landmarks).reshape(1, -1)

                # Predict gesture
                prediction = model.predict(input_data)[0]

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

                    print("❌ Wrong gesture!")
                    print(f"❤️ Lives remaining: {lives}")

                    if lives <= 0:

                        completed = True

        # ==========================================
        # CAMERA DISPLAY
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
            f"{mission}",
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

        # Display camera
        cv2.imshow(
            "CUES & CLUES - Mission",
            frame
        )

        # Q = Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            lives = 0
            completed = True
            break

    # Close camera
    cap.release()
    cv2.destroyAllWindows()

    # ==========================================
    # MISSION RESULT
    # ==========================================

    if correct_frames >= 10:

        print("✅ Mission completed!")
        print(f"⭐ Score: {score}")
        print(f"❤️ Lives: {lives}")

    elif lives <= 0:

        print("💔 GAME OVER!")

    else:

        print("❌ Mission failed.")

# ==========================================
# FINAL RESULT
# ==========================================

print("\n================================")

if lives > 0:

    print("       🎉 GAME COMPLETED!")

else:

    print("       💔 GAME OVER!")

print("================================")
print(f"⭐ Final Score: {score}")
print(f"❤️ Lives Remaining: {lives}")
print("================================")

hands.close()