# MemmoVisio_refactor.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import pydicom
from streamlit_chat import message
import requests
import time
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import streamlit.components.v1 as components
import plotly.graph_objects as go
import io

# -------------------------
# Configuration & secrets
# -------------------------
st.set_page_config(page_title="MammoCare", page_icon="🩺", layout="wide")

API_KEY = st.secrets.get("AIXPLAIN_API_KEY")
API_URL = "https://models.aixplain.com/api/v1/execute/6414bd3cd09663e9225130e8"
headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

# -------------------------
# Helpers & caching
# -------------------------
@st.cache_data
def load_csv_data(path="treatment_centers.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()  # empty placeholder
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Unable to load CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def load_dicom_file(file_obj):
    try:
        dcm = pydicom.dcmread(file_obj)
        img = dcm.pixel_array.astype(np.float32)
        if img.max() > 255:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)
    except Exception as e:
        st.error(f"Error reading DICOM: {e}")
        return None

def preprocess_image(image, blur_kernel_size=5):
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(image)
    k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    blurred = cv2.GaussianBlur(equalized, (k, k), 0)
    return blurred

def remove_pectoral_muscle(image, side, start, end, thickness=6):
    # start/end are (x,y) tuples
    if image is None:
        return None
    x1, y1 = start
    x2, y2 = end
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=thickness)
    h, w = image.shape[:2]

    if side == "Left":
        if y1 < y2:
            points = np.array([[0,0],[0,h-1],[y2,x2],[y1,x1]])
        else:
            points = np.array([[y1,x1],[y2,x2],[w-1,h-1],[w-1,0]])
    else:  # Right
        if y1 < y2:
            points = np.array([[y1,x1],[y2,x2],[w-1,h-1],[w-1,0]])
        else:
            points = np.array([[0,0],[0,h-1],[y2,x2],[y1,x1]])

    cv2.fillPoly(mask, [points], 255)
    out = image.copy()
    out[mask == 255] = 0
    return out

def find_highest_dense_region(image):
    if image is None:
        return None, None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return thresh, np.stack([gray]*3, axis=-1)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    highest = cv2.bitwise_and(gray, gray, mask=mask)
    dense_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dense_vis, [largest], -1, (0,0,255), 2)
    return thresh, dense_vis

def poll_aixplain_model(request_id):
    poll_url = f"https://models.aixplain.com/api/v1/data/{request_id}"
    for _ in range(60):  # timeout after ~5 minutes (60 * 5s)
        r = requests.get(poll_url, headers=headers)
        if r.status_code == 200:
            j = r.json()
            if j.get("completed", False):
                return j.get("data", "No result data available")
            time.sleep(5)
        else:
            return f"Polling failed: {r.status_code} {r.text}"
    return "Timeout while waiting for model."

def query_aixplain_model(user_input):
    if not API_KEY:
        return "AI model disabled — AIXPLAIN_API_KEY not configured in secrets."
    payload = {"text": user_input}
    try:
        r = requests.post(API_URL, headers={**headers, "Content-Type":"application/json"}, json=payload)
        if r.status_code in (200,201):
            j = r.json()
            req_id = j.get("requestId") or j.get("id")
            if req_id:
                return poll_aixplain_model(req_id)
            return j
        return f"API error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Exception calling AI: {e}"

# -------------------------
# Load CSV data
# -------------------------
data = load_csv_data()

# -------------------------
# Sidebar / Layout
# -------------------------
with st.sidebar:
    selected = st.radio("Main Menu", ["Home", "MammoVision", "MP Muscle Removal", "AP Muscle Removal", "Treatment Centers", "How to Use This Software", "Presentation Deck", "Contact Us"])

st.markdown(
    """
    <style>
      .stAppHeader { background-color:#31333f !important; color:white !important; padding:6px; }
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.header("MammoCare 🩺")
st.caption("AI-assisted mammogram pre-processing — simplified UI")

# -------------------------
# HOME
# -------------------------
if selected == "Home":
    st.subheader("Simplifying Breast Cancer Visualization")
    st.markdown(
        """
        MammoCare helps remove artifacts (like pectoral muscle) from mammograms and highlights dense regions.
        This simplified UI is optimized for web deployment (Streamlit Cloud).
        """
    )

# -------------------------
# MP MANUAL MUSCLE REMOVAL
# -------------------------
if selected == "MP Muscle Removal":
    st.title("Manual Pectoral Muscle Removal")
    st.info("Upload JPG/PNG or DICOM (.dcm). Adjust the coordinates and side, then process.")
    uploaded = st.file_uploader("Upload mammogram (dcm/jpg/png)", type=["dcm","jpg","jpeg","png"])
    blur_k = st.slider("Gaussian blur kernel (odd recommended)", 1, 21, 5, 2)
    side = st.selectbox("Breast side", ["Left","Right"])

    if uploaded:
        # load image (DICOM or image)
        if uploaded.name.lower().endswith(".dcm"):
            img = load_dicom_file(uploaded)
        else:
            pil = Image.open(uploaded).convert("L")
            img = np.array(pil)

        st.session_state["orig_image"] = img
        st.image(img, caption="Original / Uploaded", use_column_width=True)

        h, w = img.shape[:2]
        st.markdown("### Define pectoral muscle line (start -> end)")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.slider("start x", 0, w-1, 0)
            x2 = st.slider("end x", 0, w-1, w-1)
        with col2:
            y1 = st.slider("start y", 0, h-1, 0)
            y2 = st.slider("end y", 0, h-1, h-1)

        pre = preprocess_image(img, blur_k)
        st.image(pre, caption="Preprocessed", use_column_width=True)

        if st.button("Remove Pectoral Muscle"):
            out = remove_pectoral_muscle(pre.copy(), side, (x1,y1), (x2,y2))
            st.session_state["out_image"] = out
            st.image(out, caption="Muscle Removed", use_column_width=True)

            thresh, dense_vis = find_highest_dense_region(out)
            st.image(thresh, caption="Thresholded (dense region mask)", use_column_width=True)
            st.image(dense_vis, caption="Highest Dense Region (highlight)", use_column_width=True)

            # download
            pil_out = Image.fromarray(out if out.ndim==2 else cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_out.save(buf, format="JPEG")
            buf.seek(0)
            st.download_button("Download Processed Image", data=buf, file_name="muscle_removed.jpg", mime="image/jpeg")

# -------------------------
# AP (auto) placeholder
# -------------------------
if selected == "AP Muscle Removal":
    st.title("Automated Pectoral Muscle Removal (Preview)")
    st.markdown("This link opens the auto removal demo (if available).")
    st.markdown('[Open Auto Pectoral Removal Demo](https://bcdauto.streamlit.app/)')

# -------------------------
# MammoVision (Plotly 3D)
# -------------------------
if selected == "MammoVision":
    st.title("MammoVision — 3D Surface / Contour Visualizations")
    uploaded_img = st.file_uploader("Upload image for 3D surface", type=["jpg","jpeg","png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("L")
        a = np.array(img) / 255.0
        x = np.linspace(0, a.shape[1]-1, a.shape[1])
        y = np.linspace(0, a.shape[0]-1, a.shape[0])
        xg, yg = np.meshgrid(x,y)
        z = a * 20.0

        fig = go.Figure(data=[go.Surface(z=z, x=xg, y=yg, showscale=False)])
        fig.update_layout(title="3D Surface of intensity", autosize=True, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Contour
        fig2 = go.Figure(data=go.Contour(z=np.flipud(a*20)))
        fig2.update_layout(title="Contour (flipped)", autosize=True, height=500)
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Treatment Centers
# -------------------------
if selected == "Treatment Centers":
    st.title("Treatment Centers")
    if data.empty:
        st.warning("treatment_centers.csv not found or empty.")
    else:
        country_filter = st.text_input("Filter by Country")
        centre_filter = st.text_input("Filter by Centre")
        town_filter = st.text_input("Filter by Town")

        mask = pd.Series(True, index=data.index)
        if country_filter:
            mask &= data["Country"].str.contains(country_filter, case=False, na=False)
        if centre_filter:
            mask &= data["Centre"].str.contains(centre_filter, case=False, na=False)
        if town_filter:
            mask &= data["Town"].str.contains(town_filter, case=False, na=False)

        filtered = data[mask]
        st.write(f"Found {len(filtered)} center(s).")
        st.dataframe(filtered, use_container_width=True)

# -------------------------
# How to Use
# -------------------------
if selected == "How to Use This Software":
    st.title("How to Use MammoCare")
    st.write("- Upload mammogram (DICOM or image).")
    st.write("- Use the Manual Muscle Removal page to remove pectoral muscle and detect dense regions.")
    st.write("- Use MammoVision for interactive 3D/contour visualizations (Plotly).")
    st.markdown("Tutorial video (optional):")
    # Provide a safe iframe only if you set an ID
    youtube_video_id = ""  # set to your YouTube ID, e.g. "9SE6B0h-4-Q"
    if youtube_video_id:
        video_file_path = f"https://www.youtube.com/embed/{youtube_video_id}?autoplay=0&mute=1"
        st.markdown(f'<iframe width="100%" height="480" src="{video_file_path}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
    else:
        st.info("No tutorial video configured. Set `youtube_video_id` in the script to embed a video.")

# -------------------------
# Presentation Deck & Contact
# -------------------------
if selected == "Presentation Deck":
    st.title("Presentation Deck")
    canva_src = "https://www.canva.com/design/DAGjBW1mHgs/EF8Q-dKJEJyFoagnUJ9IsA/view?embed"
    st.markdown(f'<iframe src="{canva_src}" style="width:100%;height:600px;border:none;"></iframe>', unsafe_allow_html=True)

if selected == "Contact Us":
    st.title("Contact")
    st.markdown("Questions? Email: support@MammoCare.com")

# -------------------------
# AI Chat (simple)
# -------------------------
st.markdown("---")
st.subheader("Ask the AI (optional)")
user_input = st.text_input("Enter your question for the aiXplain model")
if st.button("Ask AI"):
    if user_input.strip():
        with st.spinner("Calling AI model..."):
            res = query_aixplain_model(user_input)
        st.write(res)
    else:
        st.error("Please enter a question.")

# Footer
st.markdown("<small>© 2025 MammoCare. Made safe for web deployment.</small>", unsafe_allow_html=True)
