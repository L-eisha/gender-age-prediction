
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Analysis App",
    page_icon="👤",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .stMetric {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #3a3a5e;
    }
    .face-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #0f3460;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-text {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .gender-badge-woman {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .gender-badge-man {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown('<p class="title-text">👤 Face Analysis App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload one or more images to detect faces and predict gender & age using deep learning.</p>', unsafe_allow_html=True)

# ── Load DNN face detector ─────────────────────────────────────
@st.cache_resource
def load_detector():
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )
    return net

net = load_detector()

# ── Helper: process one image ──────────────────────────────────
def process_image(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    results = []
    output_image = image.copy()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")

            pad = 20
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            face_crop = image[y1p:y2p, x1p:x2p]

            try:
                result = DeepFace.analyze(
                    face_crop,
                    actions=["age", "gender"],
                    enforce_detection=False
                )
                age         = result[0]["age"]
                gender      = result[0]["dominant_gender"]
                gender_conf = result[0]["gender"][gender]

                results.append({
                    "gender": gender,
                    "gender_conf": gender_conf,
                    "age": age,
                    "det_conf": confidence * 100,
                    "crop": face_crop
                })

                # Draw on image
                color = (240, 100, 200) if gender == "Woman" else (100, 200, 240)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_image,
                    f"{gender} ({gender_conf:.0f}%)  Age: {age}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            except Exception as e:
                pass

    return output_image, results

# ── File uploader (multiple files) ────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.markdown(f"### 📸 {uploaded_file.name}")

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original**")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            output_image, results = process_image(image)

        with col2:
            st.markdown("**Detected**")
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download button
        result_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        st.download_button(
            label=f"⬇️ Download result — {uploaded_file.name}",
            data=buf.getvalue(),
            file_name=f"result_{uploaded_file.name}",
            mime="image/jpeg"
        )

        # Result cards
        if results:
            st.markdown(f"**{len(results)} face(s) found**")
            cols = st.columns(len(results))
            for idx, (r, col) in enumerate(zip(results, cols)):
                with col:
                    badge_class = "gender-badge-woman" if r["gender"] == "Woman" else "gender-badge-man"
                    st.markdown(f'<span class="{badge_class}">{r["gender"]}</span>', unsafe_allow_html=True)
                    st.metric("Age", f"{r['age']} yrs")
                    st.metric("Gender confidence", f"{r['gender_conf']:.1f}%")
                    st.metric("Face detection", f"{r['det_conf']:.1f}%")
        else:
            st.warning("No faces detected in this image.")

else:
    st.info("👆 Upload one or more images to get started!")