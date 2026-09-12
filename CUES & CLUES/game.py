import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
from urllib.request import urlretrieve


# =========================================================
# CUES & CLUES
# AI-POWERED SIGN LANGUAGE COMMUNICATION GAME
# =========================================================

st.set_page_config(
    page_title="CUES & CLUES",
    page_icon="🤟",
    layout="centered"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "knn_model.pkl"

HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# =========================================================
# DOWNLOAD MEDIAPIPE HAND LANDMARK MODEL
# =========================================================

@st.cache_resource
def download_hand_model():

    if HAND_MODEL_PATH.exists():
        return HAND_MODEL_PATH

    try:
        urlretrieve(
            HAND_MODEL_URL,
            HAND_MODEL_PATH
        )

    except Exception as e:

        st.error(
            "❌ Could not download the MediaPipe hand model."
        )

        st.code(str(e))

        st.stop()

    return HAND_MODEL_PATH


# =========================================================
# LOAD KNN MODEL
# =========================================================

@st.cache_resource
def load_knn_model():

    if not MODEL_PATH.exists():

        st.error(
            f"❌ KNN model not found:\n{MODEL_PATH}"
        )

        st.stop()

    try:

        model = joblib.load(
            MODEL_PATH
        )

        return model

    except Exception as e:

        st.error(
            "❌ Could not load the KNN AI model."
        )

        st.code(str(e))

        st.stop()


# =========================================================
# CREATE MEDIAPIPE HAND LANDMARKER
# =========================================================

@st.cache_resource
def create_hand_landmarker():

    hand_model = download_hand_model()

    try:

        BaseOptions = mp.tasks.BaseOptions

        HandLandmarker = (
            mp.tasks.vision.HandLandmarker
        )

        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=str(
                    hand_model
                )
            ),

            running_mode=RunningMode.IMAGE,

            num_hands=1,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5
        )

        landmarker = (
            HandLandmarker.create_from_options(
                options
            )
        )

        return landmarker

    except Exception as e:

        st.error(
            "❌ Could not start MediaPipe Hand Landmarker."
        )

        st.code(str(e))

        st.stop()


# =========================================================
# LOAD AI COMPONENTS
# =========================================================

model = load_knn_model()

hand_landmarker = create_hand_landmarker()


# =========================================================
# GAME MISSIONS
# =========================================================

MISSIONS = [

    (
        "Mission 1",
        "You are thirsty. Show WATER.",
        "WATER"
    ),

    (
        "Mission 2",
        "You need assistance. Show HELP.",
        "HELP"
    ),

    (
        "Mission 3",
        "Answer the question. Show YES.",
        "YES"
    )

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


# =========================================================
# TITLE
# =========================================================

st.title("🤟 CUES & CLUES")

st.subheader(
    "AI-Powered Sign Language Communication Game"
)

st.write(
    "Complete each mission using sign language."
)


# =========================================================
# START SCREEN
# =========================================================

if not st.session_state.game_started:

    st.info(
        "🎯 Complete 3 communication missions."
    )

    st.markdown(
        "### 🎮 Game Rules"
    )

    st.write(
        "❤️ Lives: 3"
    )

    st.write(
        "⭐ Correct mission: +10 points"
    )

    st.write(
        "🤟 Show the requested sign clearly."
    )

    st.write(
        "📷 Use your camera to submit your sign."
    )

    st.markdown("---")

    if st.button(
        "🚀 START GAME",
        use_container_width=True
    ):

        st.session_state.game_started = True

        st.session_state.mission_index = 0

        st.session_state.score = 0

        st.session_state.lives = 3

        st.rerun()

    st.stop()


# =========================================================
# GAME OVER
# =========================================================

if st.session_state.lives <= 0:

    st.error(
        "💀 GAME OVER"
    )

    st.markdown(
        f"## Final Score: {st.session_state.score}"
    )

    st.write(
        "You ran out of lives."
    )

    st.markdown("---")

    if st.button(
        "🔄 PLAY AGAIN",
        use_container_width=True
    ):

        st.session_state.game_started = False

        st.session_state.mission_index = 0

        st.session_state.score = 0

        st.session_state.lives = 3

        st.rerun()

    st.stop()


# =========================================================
# GAME COMPLETED
# =========================================================

if (
    st.session_state.mission_index
    >= len(MISSIONS)
):

    st.success(
        "🏆 GAME COMPLETED!"
    )

    st.balloons()

    st.markdown(
        f"## ⭐ Final Score: "
        f"{st.session_state.score}"
    )

    st.write(
        f"❤️ Lives Remaining: "
        f"{st.session_state.lives}"
    )

    st.write(
        "🎉 Excellent communication!"
    )

    st.markdown("---")

    if st.button(
        "🔄 PLAY AGAIN",
        use_container_width=True
    ):

        st.session_state.game_started = False

        st.session_state.mission_index = 0

        st.session_state.score = 0

        st.session_state.lives = 3

        st.rerun()

    st.stop()


# =========================================================
# CURRENT MISSION
# =========================================================

mission, instruction, correct_gesture = (
    MISSIONS[
        st.session_state.mission_index
    ]
)


# =========================================================
# SCOREBOARD
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "⭐ SCORE",
        st.session_state.score
    )


with col2:

    st.metric(
        "❤️ LIVES",
        st.session_state.lives
    )


with col3:

    st.metric(
        "🎯 MISSION",
        f"{st.session_state.mission_index + 1}/3"
    )


# =========================================================
# MISSION DISPLAY
# =========================================================

st.markdown("---")

st.header(
    f"🎯 {mission}"
)

st.markdown(
    f"### {instruction}"
)

st.success(
    f"🤟 Required sign: **{correct_gesture}**"
)

st.markdown("---")


# =========================================================
# CAMERA
# =========================================================

picture = st.camera_input(
    "📷 Take a picture of your sign"
)


# =========================================================
# CAMERA PROCESSING
# =========================================================

if picture is not None:

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

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

        st.error(
            "❌ Could not read the camera image."
        )

        st.stop()


    # -----------------------------------------------------
    # CONVERT BGR TO RGB
    # -----------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # -----------------------------------------------------

    try:

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

    except Exception as e:

        st.error(
            "❌ Could not create MediaPipe image."
        )

        st.code(str(e))

        st.stop()


    # -----------------------------------------------------
    # DETECT HAND
    # -----------------------------------------------------

    try:

        result = hand_landmarker.detect(
            mp_image
        )

    except Exception as e:

        st.error(
            "❌ Hand detection failed."
        )

        st.code(str(e))

        st.stop()


    # =====================================================
    # HAND DETECTED
    # =====================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # -------------------------------------------------
        # HAND CONNECTIONS
        # -------------------------------------------------

        connections = [

            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (0, 17)
        ]


        height, width = (
            frame.shape[:2]
        )


        # -------------------------------------------------
        # DRAW LANDMARK POINTS
        # -------------------------------------------------

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # -------------------------------------------------
        # DRAW CONNECTIONS
        # -------------------------------------------------

        for start, end in connections:

            x1 = int(
                hand[start].x * width
            )

            y1 = int(
                hand[start].y * height
            )

            x2 = int(
                hand[end].x * width
            )

            y2 = int(
                hand[end].y * height
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # =================================================
        # EXTRACT LANDMARK FEATURES
        # =================================================

        landmarks = []

        for landmark in hand:

            landmarks.append(
                landmark.x
            )

            landmarks.append(
                landmark.y
            )

            landmarks.append(
                landmark.z
            )


        input_data = np.array(
            landmarks,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )


        # =================================================
        # AI PREDICTION
        # =================================================

        try:

            prediction = model.predict(
                input_data
            )[0]

            prediction = str(
                prediction
            ).upper().strip()

        except Exception as e:

            st.error(
                "❌ The AI model could not make a prediction."
            )

            st.code(str(e))

            st.stop()


        # =================================================
        # DISPLAY DETECTED HAND
        # =================================================

        st.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            caption="🤟 Hand detected",
            use_container_width=True
        )


        # =================================================
        # SHOW AI RESULT
        # =================================================

        st.markdown(
            f"### 🤖 Detected sign: **{prediction}**"
        )


        # =================================================
        # CORRECT ANSWER
        # =================================================

        if prediction == correct_gesture:

            st.success(
                f"✅ CORRECT! "
                f"You showed **{correct_gesture}**."
            )

            st.session_state.score += 10

            st.session_state.mission_index += 1

            st.balloons()

            st.rerun()


        # =================================================
        # WRONG ANSWER
        # =================================================

        else:

            st.warning(
                f"❌ The AI detected **{prediction}**."
            )

            st.info(
                f"Try showing **{correct_gesture}**."
            )

            st.write(
                "📷 Take another picture when ready."
            )


    # =====================================================
    # NO HAND DETECTED
    # =====================================================

    else:

        st.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            caption="No hand detected",
            use_container_width=True
        )

        st.warning(
            "✋ No hand detected."
        )

        st.info(
            "Place your hand clearly inside "
            "the camera frame and try again."
        )


# =========================================================
# CAMERA INSTRUCTIONS
# =========================================================

else:

    st.info(
        "📷 Click the camera button above, "
        "allow camera access, and show your sign."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "CUES & CLUES • AI-powered Sign Language Communication"
)
