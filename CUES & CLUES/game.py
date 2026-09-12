import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path

# =========================================================
# CUES & CLUES - Browser Sign Language Game
# =========================================================

st.set_page_config(
    page_title="CUES & CLUES",
    page_icon="🤟",
    layout="centered"
)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the AI model: {e}")
    st.stop()

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================================================
# GAME DATA
# =========================================================

MISSIONS = [
    ("Mission 1", "You are thirsty. Show WATER.", "WATER"),
    ("Mission 2", "You need assistance. Show HELP.", "HELP"),
    ("Mission 3", "Answer the question. Show YES.", "YES"),
]

# =========================================================
# SESSION STATE
# =========================================================

if "game_started" not in st.session_state:
    st.session_state.game_started = False

if "mission_index" not in st.session_state:
    st.session_state.mission_index = 0

if "score" not in st.session_state:
    st.session_state.score = 0

if "lives" not in st.session_state:
    st.session_state.lives = 3

if "message" not in st.session_state:
    st.session_state.message = ""

# =========================================================
# TITLE
# =========================================================

st.title("🤟 CUES & CLUES")
st.subheader("Sign Language Communication Game")

st.write(
    "Use your hand signs to complete each mission!"
)

# =========================================================
# START SCREEN
# =========================================================

if not st.session_state.game_started:

    st.info(
        "🎯 Complete 3 missions using sign language."
    )

    st.write("### Game Rules")
    st.write("❤️ Lives: 3")
    st.write("⭐ Each completed mission: +10 points")
    st.write("🤟 Hold the correct sign to complete a mission.")

    if st.button("🚀 START GAME", use_container_width=True):

        st.session_state.game_started = True
        st.session_state.mission_index = 0
        st.session_state.score = 0
        st.session_state.lives = 3
        st.rerun()

    st.stop()

# =========================================================
# GAME OVER CHECK
# =========================================================

if st.session_state.lives <= 0:

    st.error("💀 GAME OVER")

    st.write(
        f"Final Score: **{st.session_state.score}**"
    )

    if st.button("🔄 PLAY AGAIN", use_container_width=True):

        st.session_state.game_started = False
        st.session_state.mission_index = 0
        st.session_state.score = 0
        st.session_state.lives = 3
        st.rerun()

    st.stop()

# =========================================================
# GAME COMPLETED CHECK
# =========================================================

if st.session_state.mission_index >= len(MISSIONS):

    st.success("🏆 GAME COMPLETED!")

    st.balloons()

    st.write(
        f"### Final Score: {st.session_state.score}"
    )

    st.write(
        f"❤️ Lives Remaining: {st.session_state.lives}"
    )

    if st.button("🔄 PLAY AGAIN", use_container_width=True):

        st.session_state.game_started = False
        st.session_state.mission_index = 0
        st.session_state.score = 0
        st.session_state.lives = 3
        st.rerun()

    st.stop()

# =========================================================
# CURRENT MISSION
# =========================================================

mission, instruction, correct_gesture = MISSIONS[
    st.session_state.mission_index
]

# =========================================================
# SCOREBOARD
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "⭐ Score",
        st.session_state.score
    )

with col2:
    st.metric(
        "❤️ Lives",
        st.session_state.lives
    )

with col3:
    st.metric(
        "🎯 Mission",
        f"{st.session_state.mission_index + 1}/3"
    )

# =========================================================
# MISSION CARD
# =========================================================

st.markdown("---")

st.header(f"🎯 {mission}")

st.write(f"### {instruction}")

st.success(
    f"Show the sign: **{correct_gesture}**"
)

st.markdown("---")

# =========================================================
# CAMERA
# =========================================================

picture = st.camera_input(
    "📷 Show your sign to the camera"
)

# =========================================================
# PROCESS IMAGE
# =========================================================

if picture is not None:

    image_bytes = picture.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error("Could not read camera image.")
        st.stop()

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # MEDIAPIPE DETECTION
    # =====================================================

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:

        results = hands.process(rgb)

    # =====================================================
    # HAND DETECTED
    # =====================================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # =================================================
        # LANDMARK FEATURES
        # =================================================

        landmarks = []

        for landmark in hand_landmarks.landmark:

            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)

        input_data = np.array(
            landmarks
        ).reshape(1, -1)

        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]

        prediction = str(prediction)

        st.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            caption="Hand detected",
            use_container_width=True
        )

        st.write(
            f"### Detected sign: **{prediction}**"
        )

        # =================================================
        # CORRECT
        # =================================================

        if prediction == correct_gesture:

            st.success(
                f"✅ CORRECT! You showed {correct_gesture}."
            )

            st.session_state.score += 10

            st.session_state.mission_index += 1

            st.balloons()

            if st.session_state.mission_index < 3:

                st.button(
                    "➡️ Continue to next mission",
                    disabled=True
                )

            st.rerun()

        # =================================================
        # WRONG
        # =================================================

        else:

            st.warning(
                f"❌ Detected {prediction}. "
                f"Try showing {correct_gesture}."
            )

            if st.button(
                "Try Again",
                use_container_width=True
            ):
                st.rerun()

    else:

        st.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            caption="No hand detected"
        )

        st.warning(
            "✋ No hand detected. "
            "Place your hand clearly in front of the camera."
        )

else:

    st.info(
        "📷 Click the camera button above and show your sign."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "CUES & CLUES • AI-powered Sign Language Communication"
)
