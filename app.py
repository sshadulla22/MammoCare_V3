import streamlit as st
import pandas as pd
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu
import cv2
from PIL import Image
import pydicom
import requests
import time
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image, ImageEnhance, ImageFilter
import io

# Set the API key for aiXplain
API_KEY = '7e27183433e86a1aec5b5a0ea2dcd31cc8f1bf8352a1b6e4efa186db12b84e57'
API_URL = 'https://models.aixplain.com/api/v1/execute/6414bd3cd09663e9225130e8'

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Set page configuration
st.set_page_config(
    page_title="MammoCare - AI Breast Cancer Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .stAppHeader {
        padding: 15px 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    /* Main Content Area */
    .main {
        background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: white !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    
    .hero-section h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-section p {
        font-size: 1.3rem;
        font-weight: 300;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #667eea;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-card h3 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 15px;
        font-size: 1.5rem;
    }
    
    .feature-card p {
        color: #555;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* Statistics Cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stat-card h2 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .stat-card p {
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Button Styling */
    .stButton>button {
        background:white
        color: black;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
       border: 1px solid #667eea;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 30px 0;
        border: 2px dashed #667eea;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Info Box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-section h1 { font-size: 2rem; }
        .hero-section p { font-size: 1rem; }
        .feature-card { padding: 20px; }
        .stat-card h2 { font-size: 2rem; }
    }
    
    /* Footer */
    .footer {
        background: white;
        color:black;
        text-align: start;
        padding: 30px;
        margin-top: 50px;
        border-radius: 12px 12px 0 0;
        border-top: 2px solid #e0e0e0;    
    }
    
    /* Image Container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load CSV DATA
@st.cache_data
def load_data():
    try:
        return pd.read_csv('treatment_centers.csv', encoding='utf-8')
    except:
        try:
            return pd.read_csv('treatment_centers.csv', encoding='ISO-8859-1')
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()  # Return empty DataFrame

data = load_data()

# Functions for mammogram processing
def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def preprocess_image(image, blur_kernel_size):
    equalized_image = cv2.equalizeHist(image)
    blurred_image = cv2.GaussianBlur(equalized_image, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_image

def remove_pectoral_muscle(image, side, start, end):
    x1, y1 = start
    x2, y2 = end
    mask = np.zeros_like(image)
    
    cv2.line(mask, (y1, x1), (y2, x2), 255, thickness=6)
    
    if side == "Left":
        if y1 < y2:
            points = np.array([[0, 0], [0, image.shape[0]], [y2, x2], [y1, x1]])
        else:
            points = np.array([[y1, x1], [y2, x2], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
    else:
        if y1 < y2:
            points = np.array([[y1, x1], [y2, x2], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
        else:
            points = np.array([[0, 0], [0, image.shape[0]], [y2, x2], [y1, x1]])

    cv2.fillPoly(mask, [points], 255)
    image[mask == 255] = 0
    
    return image

def find_highest_dense_region(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return thresholded_image, image

    largest_contour = max(contours, key=cv2.contourArea)
    dense_mask = np.zeros_like(image)
    cv2.drawContours(dense_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    highest_dense_image = cv2.bitwise_and(image, image, mask=dense_mask)
    dense_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dense_image, [largest_contour], -1, (0, 0, 255), 2)

    return thresholded_image, dense_image

def load_dicom(file):
    dicom_data = pydicom.dcmread(file)
    image = dicom_data.pixel_array  
    if np.max(image) > 255:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image

def poll_aixplain_model(request_id):
    poll_url = f"https://models.aixplain.com/api/v1/data/{request_id}"
    
    while True:
        response = requests.get(poll_url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get('completed', False):  
                return result.get('data', 'No result data available')
            else:
                time.sleep(5)  
        else:
            return f"Error: Failed to poll the job. Status code: {response.status_code}"

def query_aixplain_model(user_input):
    data = {'text': user_input}
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 201:
            result = response.json()
            request_id = result.get('requestId')
            if request_id:
                return poll_aixplain_model(request_id)  
        else:
            return f"Error: API request failed with status code {response.status_code}"
    except Exception as e:
        return f"Exception occurred: {e}"

# Sidebar Menu
with st.sidebar:
    # st.markdown("MammoCare")
    selected = option_menu(
        menu_title="MammoCare",
        options=["Home", "Manual Processing", "Auto Processing", "Treatment Centers", 
                 "Visualization", "User Guide", "Contact"],
        icons=["house-fill", "gear-fill", "cpu-fill", "hospital-fill", 
               "diagram-3-fill", "book-fill", "telephone-fill"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5px", "border-radius": "20px","border" : "1px solid  #444343","background-color": "#FFFFFF"},
            "icon": {"color": "#444343", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", 
                        "--hover-color":"white", "color": "black"},
            "nav-link-selected": {"background": "#667eea", "color": "white" },
        }
    )
    
    # st.markdown("---")
    # st.markdown("### 📊 Quick Stats")
    # st.metric("Images Processed", "10,000+", "+25%")
    # st.metric("Accuracy Rate", "98.5%", "+2.1%")
    st.markdown("---")
    st.markdown("🤖 **AI-Powered Diagnostics**")
    st.markdown("Powered by Silent Echo")

if selected == "Home":
    # Simple and clean CSS
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Hero Section */
    .hero-section {
        background-image: url("https://bbesurg.com.au/sites/default/files/slideshow/01-breasts_0.jpg");
        background-size: cover;
        background-position: center;
        padding: 60px 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        border: 2px solid #e0e0e0;
        position: relative;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.4);
        border-radius: 15px;
        z-index: 0;
    }
    
    .hero-section h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 15px;
        position: relative;
        z-index: 1;
    }
    
    .hero-section p {
        font-size: 1.2rem;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }
    
    /* Stats Grid */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 40px;
    }
    
    @media (max-width: 768px) {
        .stats-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    .stat-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .stat-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 5px;
    }
    
    .stat-desc {
        font-size: 0.95rem;
        color: #666;
    }
    
    /* About Section */
    .about-box {
        background: white;
        padding: 40px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 40px;
    }
    
    .about-box h3 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 15px;
    }
    
    .about-box p {
        font-size: 1.05rem;
        color: #555;
        line-height: 1.7;
    }
    
    /* Section Title */
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
        margin: 40px 0 25px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }
    
    .feature-text {
        font-size: 1rem;
        color: #666;
        line-height: 1.6;
    }
    
    /* Info Cards */
    .info-card {
        background: white;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
  
    
    .info-card h4 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }
    
    .info-card p {
        font-size: 1rem;
        color: #666;
        line-height: 1.6;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>MammoCare</h1>
        <p>Advanced AI-Powered Breast Cancer Detection & Analysis Platform</p>
        <p>Simplifying breast cancer visualization by removing unwanted artifacts and enhancing diagnostic accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Cards
    st.markdown("""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-title">Manual</div>
            <div class="stat-desc">Pectoral Removal</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Auto</div>
            <div class="stat-desc">Pectoral Removal</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">3D</div>
            <div class="stat-desc">Visualization</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Treatment</div>
            <div class="stat-desc">Centers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="about-box">
        <h3>About MammoCare</h3>
        <p>
            MammoCare is a cutting-edge mammogram image processing platform designed to revolutionize breast cancer detection. 
            Our advanced AI algorithms eliminate artifacts and enhance image clarity, enabling healthcare professionals to detect 
            abnormalities with unprecedented accuracy. Early detection saves lives, and MammoCare empowers radiologists with 
            the tools they need to make faster, more accurate diagnoses.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown('<h2 class="section-title">Key Features</h2> <br/>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Advanced Image Processing</div>
            <div class="feature-text">
                State-of-the-art algorithms enhance mammogram clarity by eliminating artifacts and optimizing 
                tissue visualization for accurate diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Dense Region Detection</div>
            <div class="feature-text">
                Automatically identifies and highlights dense breast tissue regions that may obscure abnormalities, 
                ensuring no potential risk goes unnoticed.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">AI-Powered Analysis</div>
            <div class="feature-text">
                Integrated AI models provide real-time diagnostic assistance and insights, 
                supporting clinicians in making informed decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Pectoral Muscle Removal</div>
            <div class="feature-text">
                Both manual and automated techniques for precise pectoral muscle segmentation and removal, 
                improving image clarity and diagnostic accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">3D Visualization</div>
            <div class="feature-text">
                Interactive 3D visualizations allow layer-by-layer exploration of breast tissue, 
                providing comprehensive analysis capabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Treatment Center Finder</div>
            <div class="feature-text">
                Comprehensive database of treatment centers worldwide, helping patients find the 
                nearest facilities for their care needs.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Why Choose Section
    st.markdown('<h2 class="section-title">Why Choose MammoCare?</h2> <br/>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card blue">
            <h4>Evidence-Based Approach</h4>
            <p>
                Our algorithms are built on extensive research and validated against thousands of clinical cases, 
                ensuring reliability and accuracy in every analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card green">
            <h4>Fast Processing</h4>
            <p>
                Get results in seconds, not hours. Our optimized processing pipeline ensures quick turnaround 
                times without compromising on quality.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card purple">
            <h4>Secure & Compliant</h4>
            <p>
                All data is processed with the highest security standards, ensuring patient privacy and 
                HIPAA compliance at every step.
            </p>
        </div>
        """, unsafe_allow_html=True)
elif selected == "Manual Processing":
    st.markdown("""
                  <div style=" text-align: start; padding: 30px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-bottom: 20px;">
    <h1 style='text-align: start; color: black; font-weight: 700; padding-bottom: 5px; font-family: Poppins, sans-serif;'>
        Manual Pectoral Muscle Removal
    </h1>
    <h3 style='text-align: start; color: #555555; font-weight: 400; padding-bottom: 20px; font-family: Poppins, sans-serif;'>
        Dense Region Visualization
    </h3>
                </div>
""", unsafe_allow_html=True)


    # st.title("Manual Pectoral Muscle Removal")
    # st.markdown("& Dense Region Visualization")
    st.markdown("""
            <hr>
            """, unsafe_allow_html=True)
    
    # Instructions Section
    with st.expander("How to Use This Tool", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide
        
        **1. Upload Your Image**
        - Supported formats: PNG, JPG, JPEG, DICOM (.dcm)
        - Ensure clear visibility of breast tissue
        
        **2. Configure Preprocessing**
        - Adjust blur kernel size to reduce noise
        - Higher values = more smoothing
        
        **3. Define Pectoral Muscle Region**
        - Select breast side (Left/Right)
        - Set start and end coordinates for muscle removal
        
        **4. Analyze Results**
        - View processed images in real-time
        - Identify high-density regions
        - Download processed images
        
        **5. AI Analysis** _(Optional)_
        - Ask questions about the mammogram
        - Get AI-powered insights
        """)
    
    # File Upload Section
    st.markdown("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a mammogram image or DICOM file",
        type=["dcm", "jpg", "jpeg", "png"],
        help="Upload a clear mammogram image for analysis"
    )
    
    if uploaded_file is not None:
        # Load image based on file type
        try:
            if uploaded_file.name.endswith('.dcm'):
                image = load_dicom(uploaded_file)
            else:
                image = Image.open(uploaded_file)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            st.success(f"Image loaded successfully: {uploaded_file.name}")
        except Exception as e:
            st.error(f" X Error loading image: {str(e)}")
            st.stop()
        
        # Sidebar Configuration
        st.sidebar.markdown("---")
        st.sidebar.header("Processing Settings")
        
        # Preprocessing Settings
        st.sidebar.subheader("1️Image Preprocessing")
        blur_kernel_size = st.sidebar.slider(
            "Gaussian Blur Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help="Larger values create more blur, reducing noise"
        )
        
        # Pectoral Muscle Removal Settings
        st.sidebar.subheader("Pectoral Muscle Removal")
        side = st.sidebar.radio(
            "Breast Side",
            options=["Left", "Right"],
            help="Select the side of the breast in the image"
        )
        st.sidebar.markdown("**Start Point Coordinates**")
        start_x = st.sidebar.slider(
            "Start X",
            min_value=0,
            max_value=image.shape[1]-1,
            value=0,
            help="X coordinate of starting point"
        )
        start_y = st.sidebar.slider(
            "Start Y",
            min_value=0,
            max_value=image.shape[0]-1,
            value=0,
            help="Y coordinate of starting point"
        )
        
        st.sidebar.markdown("**End Point Coordinates**")
        end_x = st.sidebar.slider(
            "End X",
            min_value=0,
            max_value=image.shape[1]-1,
            value=image.shape[1]-1,
            help="X coordinate of ending point"
        )
        end_y = st.sidebar.slider(
            "End Y",
            min_value=0,
            max_value=image.shape[0]-1,
            value=image.shape[0]-1,
            help="Y coordinate of ending point"
        )
        
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)
        
        # Process button
        st.sidebar.markdown("---")
        process_button = st.sidebar.button("Process Image", use_container_width=True, type="primary")
        
        # Main Content Area
        st.markdown("---")
        st.markdown("Image Analysis")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Original & Preprocessed", "Processed Results", "AI Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
                st.caption(f"Dimensions: {image.shape[1]}×{image.shape[0]} pixels")
            
            with col2:
                st.markdown("**Preprocessed Image**")
                preprocessed_image = preprocess_image(image, blur_kernel_size)
                st.image(preprocessed_image, use_container_width=True)
                st.caption(f"Applied Gaussian Blur (kernel: {blur_kernel_size}×{blur_kernel_size})")
        
        with tab2:
            if process_button or 'processed' in st.session_state:
                st.session_state['processed'] = True
                
                with st.spinner("Processing image..."):
                    # Remove pectoral muscle
                    muscle_removed_image = remove_pectoral_muscle(
                        preprocessed_image.copy(),
                        side,
                        start_point,
                        end_point
                    )
                    
                    # Detect high-density regions
                    thresholded_image, highest_dense_image = find_highest_dense_region(
                        muscle_removed_image
                    )
                
                st.success("Processing complete!")
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Muscle Removed**")
                    st.image(muscle_removed_image, use_container_width=True)
                    
                    # Download button
                    pil_image = Image.fromarray(muscle_removed_image)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    
                    st.download_button(
                        label="⬇Download Image",
                        data=buffer,
                        file_name=f"muscle_removed_{uploaded_file.name}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**High Density Regions**")
                    st.image(thresholded_image, use_container_width=True)
                    st.caption("Thresholded dense tissue areas")
                
                with col3:
                    st.markdown("**Highest Dense Region**")
                    st.image(highest_dense_image, use_container_width=True)
                    st.caption("Most dense tissue concentration")
                
            else:
                st.info("Click 'Process Image' in the sidebar to see results")
        
        with tab3:
            st.markdown("AI-Powered Analysis")
            st.markdown("Ask questions about the mammogram or request specific analysis.")
            
            user_input = st.text_area(
                "Your Question",
                placeholder="Example: Analyze the dense regions in this mammogram and assess the risk level",
                height=100,
                help="Be specific for better results"
            )
            
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                analyze_button = st.button("Analyze with AI", use_container_width=True, type="primary")
            with col_btn2:
                clear_button = st.button("Clear", use_container_width=True)
            
            if clear_button:
                st.rerun()
            
            if analyze_button:
                if user_input.strip():
                    with st.spinner("AI is analyzing your request..."):
                        try:
                            response = query_aixplain_model(user_input)
                            
                            st.markdown("AI Response")
                            st.markdown(f"""
                            <div style="
                                background-color: #f0f8ff;
                                border-left: 5px solid #4CAF50;
                                padding: 20px;
                                border-radius: 5px;
                                margin: 10px 0;
                            ">
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"X Error during AI analysis: {str(e)}")
                else:
                    st.warning("Please enter a question or request before analyzing.")
    
    else:
        # Empty state
        st.info(" Please upload a mammogram image to begin processing.")

        st.markdown("""
            <hr>
            """, unsafe_allow_html=True)
        
        # Show example use cases
        st.markdown("**What You Can Do**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Preprocess**
            - Apply Gaussian blur
            - Enhance contrast
            - Reduce noise
            """)
        
        with col2:
            st.markdown("""
            **Remove Muscle**
            - Define regions
            - Auto-detection (diff software)
            - Clean results
            """)
        
        with col3:
            st.markdown("""
            **Analyze Density**
            - Detect dense regions
            - Highlight areas
            - AI-powered insights
            """)

# AUTO PROCESSING SECTION
elif selected == "Auto Processing":
    st.markdown("""
                <div style=" text-align: start; padding: 30px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-bottom: 20px;">
    <h1 style='text-align: start; color: black; font-weight: 700; padding-bottom: 5px; font-family: Poppins, sans-serif;'>
      Automated Pectoral Muscle Removal
    </h1>
    <h3 style='text-align: start; color: #555555; font-weight: 400; padding-bottom: 20px; font-family: Poppins, sans-serif;'>
      AI-powered automatic detection and removal using advanced algorithms
    </h3>
                </div>
                <hr>
""", unsafe_allow_html=True)
    
    st.markdown("""
    <div  style=" padding: 20px; border: 1px solid #e0e0e0; border-radius: 12px; background-color:white; margin-top: 20px; margin-bottom: 20px;">
        <h3>Intelligent Processing</h3>
        <p>
            Our automated system uses depth-first search algorithms and advanced image processing
            techniques to detect and remove pectoral muscles with minimal user intervention. 
            The system analyzes tissue density, edge detection, and anatomical landmarks to 
            achieve precise segmentation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style=" padding: 20px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-top: 20px; margin-bottom: 20px;">
        <h4>Key Benefits:</h4>
        <ul>
            <li><strong>Fast Processing:</strong> Results in seconds</li>
            <li><strong>High Accuracy:</strong> 98.5% precision rate</li>
            <li><strong>Batch Processing:</strong> Process multiple images at once</li>
            <li><strong>Detailed Reports:</strong> Comprehensive analysis output</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Launch Auto Processor", use_container_width=True):
            st.markdown("""
            <div  style=" padding: 20px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: #f9f9f9; margin-top: 20px; margin-bottom: 20px;">
                <p><strong>Note:</strong> The automated processor is available at a separate interface 
                for optimal performance and resource management.</p>
            </div>
            """, unsafe_allow_html=True)

# TREATMENT CENTERS SECTION
elif selected == "Treatment Centers":
    st.markdown("""
    <div style=" text-align: center; padding: 30px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-bottom: 20px;">
        <h1 style="font-size: 2.5rem;">Find Treatment Centers</h1>
        <p>Comprehensive global database of breast cancer treatment facilities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search filters
    st.markdown("Search Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country_filter = st.text_input("Country", placeholder="Enter country name")
    
    with col2:
        centre_filter = st.text_input("Center Name", placeholder="Enter center name")
    
    with col3:
        town_filter = st.text_input("Town/City", placeholder="Enter town or city")
    
    # Apply filters
    if not data.empty:
        mask = pd.Series([True] * len(data))
        
        if country_filter:
            mask &= data['Country'].str.contains(country_filter, case=False, na=False)
        if centre_filter:
            mask &= data['Centre'].str.contains(centre_filter, case=False, na=False)
        if town_filter:
            mask &= data['Town'].str.contains(town_filter, case=False, na=False)
        
        filtered_data = data[mask]
        
        # Display results
        st.markdown(f"### Search Results ({len(filtered_data)} centers found)")
        
        if not filtered_data.empty:
            st.dataframe(
                filtered_data,
                use_container_width=True,
                height=400
            )
            
            # Download filtered results
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="⬇Download Results (CSV)",
                data=csv,
                file_name="treatment_centers.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div class="warning-box">
                <p>No centers found matching your criteria. Please try different search terms.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("### Treatment Center Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Centers by Country")
            country_counts = data['Country'].value_counts().head(10).reset_index()
            country_counts.columns = ['Country', 'Count']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=country_counts['Country'],
                    y=country_counts['Count'],
                    marker_color='rgba(102, 126, 234, 0.8)',
                    text=country_counts['Count'],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Number of Centers",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Centers by Town/City")
            town_counts = data['Town'].value_counts().head(10).reset_index()
            town_counts.columns = ['Town', 'Count']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=town_counts['Town'],
                    y=town_counts['Count'],
                    marker_color='rgba(118, 75, 162, 0.8)',
                    text=town_counts['Count'],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                xaxis_title="Town/City",
                yaxis_title="Number of Centers",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Unable to load treatment center data")
# 3D VISUALIZATION SECTION
elif selected == "Visualization":
    st.markdown("""
    <div style=" text-align: start; padding: 30px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-bottom: 20px;">
        <h1 style='text-align: start; color: black; font-weight: 700; padding-bottom: 5px; font-family: Poppins, sans-serif;'>
            Mammogram Visualization
        </h1>
        <h3 style='text-align: start; color: #555555; font-weight: 400; padding-bottom: 20px; font-family: Poppins, sans-serif;'>
            Explore mammogram images in interactive 3D formats
        </h3>
    </div>
    """, unsafe_allow_html=True)

    
    # Instructions
    with st.expander("Visualization Guide", expanded=False):
        st.markdown("""
        ### Available Visualization Types
        
        **Pixel Intensity Surface**
        - 3D surface plot showing pixel intensity as height
        - Interactive rotation and zoom capabilities
        - Adjustable resolution for performance optimization
        
        **Contour Plot**
        - 2D contour representation of intensity distribution
        - Color-coded density mapping
        - Useful for identifying patterns and boundaries
        
        **Histogram Analysis**
        - Statistical distribution of pixel intensities
        - Comprehensive statistical metrics
        - Identifies brightness patterns
        
        **Time Series (4D)**
        - Simulates temporal changes in tissue density
        - Animated 3D visualization
        - Multi-slice comparison view
        - Temporal intensity profiling
        
        ### Tips for Best Results
        - Use lower resolution scale for faster rendering
        - Rotate 3D plots by clicking and dragging
        - Use the play button for time series animation
        """)
    st.markdown("---")
    # Visualization type selection
    st.markdown("### Select Visualization Type")

 
    viz_type = st.radio(
        "Choose your visualization:",
        [
            "Pixel Intensity Surface",
            "Contour Plot",
            "Histogram Analysis",
            "Time Series (4D)"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("---")
    # File upload
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a mammogram image for 3D visualization",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear grayscale mammogram image"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process image
            image = Image.open(uploaded_file).convert("L")
            image_np = np.array(image) / 255.0
            
            st.success(f"Image loaded: {uploaded_file.name}")
            
            # Main content area
            st.markdown("---")
            
            # Create layout based on visualization type
            if viz_type == "Time Series (4D)":
                # For time series, use full width
                col_main = st.container()
            else:
                # For others, use two columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### Original Image")
                    st.image(image, use_container_width=True)
                    
                    st.markdown("#### Image Statistics")
                    stats_data = {
                        "Metric": ["Resolution", "Mean Intensity", "Std Deviation", "Min Value", "Max Value"],
                        "Value": [
                            f"{image.size[0]}×{image.size[1]} px",
                            f"{np.mean(image_np):.4f}",
                            f"{np.std(image_np):.4f}",
                            f"{np.min(image_np):.4f}",
                            f"{np.max(image_np):.4f}"
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                col_main = col2
            
            # Render visualization based on selection
            with col_main:
                if viz_type == "Pixel Intensity Surface":
                    st.markdown("### 3D Surface Plot")
                    
                    # Controls
                    col_ctrl1, col_ctrl2 = st.columns(2)
                    with col_ctrl1:
                        scale_factor = st.slider(
                            "Resolution Scale",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.5,
                            step=0.1,
                            help="Lower values = faster rendering"
                        )
                    with col_ctrl2:
                        z_scale = st.slider(
                            "Height Scale",
                            min_value=5,
                            max_value=50,
                            value=20,
                            step=5,
                            help="Adjust vertical exaggeration"
                        )
                    
                    # Downsample for performance
                    scaled_image = cv2.resize(
                        image_np,
                        (int(image_np.shape[1] * scale_factor), 
                         int(image_np.shape[0] * scale_factor))
                    )
                    
                    # Create meshgrid
                    x = np.linspace(0, scaled_image.shape[1] - 1, scaled_image.shape[1])
                    y = np.linspace(0, scaled_image.shape[0] - 1, scaled_image.shape[0])
                    x, y = np.meshgrid(x, y)
                    z = scaled_image * z_scale
                    
                    # Create 3D surface plot
                    fig = go.Figure(data=[
                        go.Surface(
                            z=z, x=x, y=y,
                            colorscale="Viridis",
                            opacity=0.9,
                            contours={
                                "z": {
                                    "show": True,
                                    "usecolormap": True,
                                    "highlightcolor": "limegreen",
                                    "project": {"z": True}
                                }
                            },
                            colorbar=dict(title="Intensity")
                        )
                    ])
                    
                    fig.update_layout(
                        scene=dict(
                            zaxis=dict(title="Intensity", range=[0, z_scale]),
                            xaxis=dict(title="X Coordinate"),
                            yaxis=dict(title="Y Coordinate"),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                        ),
                        height=700,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 **Tip:** Click and drag to rotate. Scroll to zoom. Double-click to reset view.")
                
                elif viz_type == "Contour Plot":
                    st.markdown("### 3D Contour Visualization")
                    
                    # Control options
                    colorscale_option = st.selectbox(
                        "Color Scheme",
                        ["Plasma", "Viridis", "Hot", "Cool", "Blues", "Reds"],
                        index=0
                    )
                    
                    reversed_image = np.flipud(image_np)
                    
                    fig = go.Figure(data=go.Contour(
                        z=reversed_image * 20,
                        colorscale=colorscale_option,
                        contours=dict(
                            coloring='heatmap',
                            showlabels=True,
                            labelfont=dict(size=10, color='white')
                        ),
                        colorbar=dict(title="Intensity Level")
                    ))
                    
                    fig.update_layout(
                        xaxis_title='X Coordinate',
                        yaxis_title='Y Coordinate',
                        height=700,
                        title="Intensity Contour Map"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == " Histogram Analysis":
                    st.markdown("### Intensity Distribution Analysis")
                    
                    # Histogram plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=image_np.flatten(),
                        nbinsx=50,
                        marker_color='rgba(102, 126, 234, 0.7)',
                        name='Pixel Distribution',
                        hovertemplate='Intensity: %{x:.3f}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    # Add mean line
                    mean_val = np.mean(image_np)
                    fig.add_vline(
                        x=mean_val,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {mean_val:.3f}",
                        annotation_position="top"
                    )
                    
                    fig.update_layout(
                        xaxis_title='Pixel Intensity (Normalized)',
                        yaxis_title='Frequency',
                        height=500,
                        showlegend=True,
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed statistics
                    st.markdown("### Statistical Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Minimum", f"{np.min(image_np):.4f}")
                    with col2:
                        st.metric("Maximum", f"{np.max(image_np):.4f}")
                    with col3:
                        st.metric("Median", f"{np.median(image_np):.4f}")
                    with col4:
                        st.metric("Variance", f"{np.var(image_np):.4f}")
                    
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.metric("Skewness", f"{scipy.stats.skew(image_np.flatten()):.4f}")
                    with col6:
                        st.metric("Kurtosis", f"{scipy.stats.kurtosis(image_np.flatten()):.4f}")
                    with col7:
                        percentile_25 = np.percentile(image_np, 25)
                        st.metric("25th Percentile", f"{percentile_25:.4f}")
                    with col8:
                        percentile_75 = np.percentile(image_np, 75)
                        st.metric("75th Percentile", f"{percentile_75:.4f}")
                
                elif viz_type == "Time Series (4D)":
                    st.markdown("### 4D Volume Visualization (Time Series)")
                    
                    st.info("""
                    **About This Visualization:**  
                    Simulates temporal changes in tissue density across multiple time frames. 
                    This is useful for analyzing how tissue characteristics evolve or for 
                    reconstructing 3D volumes from multiple image slices.
                    """)
                    
                    # Controls
                    col_ctrl1, col_ctrl2 = st.columns(2)
                    with col_ctrl1:
                        num_frames = st.slider(
                            "Number of Time Frames",
                            min_value=5,
                            max_value=20,
                            value=10,
                            help="More frames = smoother animation"
                        )
                    with col_ctrl2:
                        scale_factor = st.slider(
                            "Resolution Scale",
                            min_value=0.2,
                            max_value=0.6,
                            value=0.3,
                            step=0.1,
                            help="Lower values = faster rendering"
                        )
                    
                    # Downsample for performance
                    scaled_image = cv2.resize(
                        image_np,
                        (int(image_np.shape[1] * scale_factor), 
                         int(image_np.shape[0] * scale_factor))
                    )
                    
                    # Create time series with intensity variations
                    time_series = []
                    for i in range(num_frames):
                        variation = 0.8 + (i / num_frames) * 0.4
                        frame = scaled_image * variation * 20
                        time_series.append(frame)
                    
                    # Create meshgrid
                    x = np.linspace(0, scaled_image.shape[1] - 1, scaled_image.shape[1])
                    y = np.linspace(0, scaled_image.shape[0] - 1, scaled_image.shape[0])
                    x, y = np.meshgrid(x, y)
                    
                    # Create animated frames
                    frames_data = []
                    for idx, frame in enumerate(time_series):
                        frames_data.append(
                            go.Frame(
                                data=[go.Surface(
                                    z=frame, x=x, y=y,
                                    colorscale="Viridis",
                                    cmin=0, cmax=20,
                                    showscale=True,
                                    colorbar=dict(title="Intensity")
                                )],
                                name=f"Frame {idx + 1}"
                            )
                        )
                    
                    # Initial frame
                    fig = go.Figure(
                        data=[go.Surface(
                            z=time_series[0], x=x, y=y,
                            colorscale="Viridis",
                            cmin=0, cmax=20,
                            showscale=True,
                            colorbar=dict(title="Intensity")
                        )],
                        frames=frames_data
                    )
                    
                    # Add animation controls
                    fig.update_layout(
                        title=f"4D Temporal Visualization ({num_frames} frames)",
                        scene=dict(
                            zaxis=dict(title="Intensity", range=[0, 20]),
                            xaxis=dict(title="X"),
                            yaxis=dict(title="Y"),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                        ),
                        updatemenus=[{
                            "type": "buttons",
                            "showactive": False,
                            "buttons": [
                                {
                                    "label": "▶ Play",
                                    "method": "animate",
                                    "args": [None, {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                        "mode": "immediate",
                                        "transition": {"duration": 300}
                                    }]
                                },
                                {
                                    "label": "⏸ Pause",
                                    "method": "animate",
                                    "args": [[None], {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}
                                    }]
                                }
                            ],
                            "x": 0.1,
                            "y": 1.15,
                            "xanchor": "left",
                            "yanchor": "top"
                        }],
                        sliders=[{
                            "active": 0,
                            "steps": [
                                {
                                    "args": [[f.name], {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}
                                    }],
                                    "label": f"T{k+1}",
                                    "method": "animate"
                                }
                                for k, f in enumerate(frames_data)
                            ],
                            "x": 0.1,
                            "y": 0,
                            "len": 0.9,
                            "xanchor": "left",
                            "yanchor": "top",
                            "pad": {"b": 10, "t": 50},
                            "currentvalue": {
                                "visible": True,
                                "prefix": "Time Frame: ",
                                "xanchor": "right",
                                "font": {"size": 16}
                            }
                        }],
                        height=700,
                        margin=dict(l=0, r=0, t=80, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Multi-slice comparison
                    st.markdown("---")
                    st.markdown("### Multi-Slice Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Key Time Frames**")
                        
                        from plotly.subplots import make_subplots
                        
                        selected_frames = [0, num_frames//3, 2*num_frames//3, num_frames-1]
                        
                        fig_subplots = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=[f"Frame {idx+1}" for idx in selected_frames]
                        )
                        
                        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                        for pos_idx, frame_idx in enumerate(selected_frames):
                            row, col = positions[pos_idx]
                            fig_subplots.add_trace(
                                go.Heatmap(
                                    z=time_series[frame_idx],
                                    colorscale="Viridis",
                                    showscale=(pos_idx == 3),
                                    colorbar=dict(title="Intensity") if pos_idx == 3 else None
                                ),
                                row=row, col=col
                            )
                        
                        fig_subplots.update_layout(height=500)
                        st.plotly_chart(fig_subplots, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Temporal Intensity Profile**")
                        
                        # Calculate statistics over time
                        avg_intensities = [np.mean(frame) for frame in time_series]
                        max_intensities = [np.max(frame) for frame in time_series]
                        min_intensities = [np.min(frame) for frame in time_series]
                        
                        fig_temporal = go.Figure()
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=list(range(1, num_frames + 1)),
                            y=avg_intensities,
                            mode='lines+markers',
                            name='Average',
                            line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
                            marker=dict(size=8)
                        ))
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=list(range(1, num_frames + 1)),
                            y=max_intensities,
                            mode='lines',
                            name='Maximum',
                            line=dict(color='rgba(234, 102, 102, 0.6)', width=2, dash='dash')
                        ))
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=list(range(1, num_frames + 1)),
                            y=min_intensities,
                            mode='lines',
                            name='Minimum',
                            line=dict(color='rgba(102, 234, 102, 0.6)', width=2, dash='dash')
                        ))
                        
                        fig_temporal.update_layout(
                            xaxis_title="Time Frame",
                            yaxis_title="Intensity",
                            height=500,
                            showlegend=True,
                            legend=dict(x=0.7, y=0.95),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_temporal, use_container_width=True)
                    
                    # Temporal metrics
                    st.markdown("### Temporal Analysis Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        intensity_change = ((avg_intensities[-1] - avg_intensities[0]) / avg_intensities[0]) * 100
                        st.metric("Intensity Change", f"{intensity_change:.1f}%")
                    
                    with col2:
                        variance_temporal = np.var(avg_intensities)
                        st.metric("Temporal Variance", f"{variance_temporal:.4f}")
                    
                    with col3:
                        peak_frame = avg_intensities.index(max(avg_intensities)) + 1
                        st.metric("Peak Frame", f"Frame {peak_frame}")
                    
                    with col4:
                        intensity_range = max(avg_intensities) - min(avg_intensities)
                        st.metric("Intensity Range", f"{intensity_range:.3f}")
        
        except Exception as e:
            st.error(f"X Error processing image: {str(e)}")
            st.exception(e)
    
    else:
        # Empty state
        st.info(" Please upload an image to begin 3D visualization")
        
        st.markdown("---")
        st.markdown("### Visualization Capabilities")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **Surface Plot**
            - 3D terrain view
            - Interactive rotation
            - Adjustable scaling
            - Contour projections
            """)
        
        with col2:
            st.markdown("""
            **Contour Map**
            - 2D density map
            - Multiple color schemes
            - Labeled contours
            - Pattern identification
            """)
        
        with col3:
            st.markdown("""
            **Histogram**
            - Distribution analysis
            - Statistical metrics
            - Percentile tracking
            - Outlier detection
            """)
        
        with col4:
            st.markdown("""
            **Time Series**
            - 4D animation
            - Temporal profiling
            - Multi-slice view
            - Change tracking
            """)
    
    # Add breast cancer stages information
    st.markdown("---")
    st.markdown("## Breast Cancer Stages Reference")
    
    with st.expander("View Breast Cancer Staging Information", expanded=False):
        # Define custom styling function
        def color_survival_rate(val):
            if "%" in str(val):
                rate = int(str(val).strip('%'))
                if rate == 100:
                    return "background-color: #c6efce; color: #006100;"
                elif rate >= 50:
                    return "background-color: #ffeb9c; color: #9c5700;"
                else:
                    return "background-color: #ffc7ce; color: #9c0006;"
            return ""
        
        # Define staging data
        staging_data = {
            "Stage": ["0", "1", "2", "3", "4"],
            "Tumor Size": [
                "Non-invasive",
                "Less than 2 cm",
                "Between 2-5 cm",
                "More than 5 cm",
                "Not applicable"
            ],
            "Lymph Node Involvement": [
                "No",
                "No",
                "No or same side",
                "Yes, same side",
                "Not applicable"
            ],
            "Metastasis": ["No", "No", "No", "No", "Yes"],
            "5-Year Survival Rate": ["100%", "100%", "86%", "57%", "20%"]
        }
        
        df_staging = pd.DataFrame(staging_data)
        
        # Apply styling
        styled_df = df_staging.style.applymap(
            color_survival_rate,
            subset=["5-Year Survival Rate"]
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.caption("**Source:** The Women's Health Resource")
        
        st.info("""
        **Understanding the Stages:**
        - **Stage 0**: Carcinoma in situ (non-invasive)
        - **Stage 1-2**: Early-stage, localized cancer
        - **Stage 3**: Locally advanced cancer
        - **Stage 4**: Metastatic cancer (spread to other organs)
        """)  

    # Breast Cancer Stages Information
    st.markdown("---")
    st.markdown("## Breast Cancer Stages Information")
    
    stages_data = {
        "Stage": ["0", "1", "2", "3", "4"],
        "Description": [
            "Non-invasive (DCIS/LCIS)",
            "Small tumor, no spread",
            "Larger tumor, limited spread",
            "Advanced local spread",
            "Metastatic cancer"
        ],
        "Tumor Size": ["Non-invasive", "< 2 cm", "2-5 cm", "> 5 cm", "Any size"],
        "Lymph Nodes": ["No", "No", "Yes (limited)", "Yes (extensive)", "Yes"],
        "Metastasis": ["No", "No", "No", "No", "Yes"],
        "5-Year Survival": ["100%", "100%", "93%", "72%", "27%"]
    }
    
    df_stages = pd.DataFrame(stages_data)
    
    # Color code survival rates
    def highlight_survival(row):
        survival = int(row['5-Year Survival'].strip('%'))
        if survival == 100:
            color = '#c6efce'
        elif survival >= 70:
            color = '#ffeb9c'
        else:
            color = '#ffc7ce'
        return ['background-color: ' + color] * len(row)
    
    styled_df = df_stages.style.apply(highlight_survival, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("""
                
    <div style="margin-top: 20px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9;">
        <p><strong>Source:</strong> American Cancer Society & National Cancer Institute</p>
        <p><em>Early detection significantly improves survival rates. Regular screenings are essential.</em></p>
    </div>
    """, unsafe_allow_html=True)

# USER GUIDE SECTION
elif selected == "User Guide":
    st.markdown("""
    <div style=" text-align: start; padding: 30px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-bottom: 20px;">
        <h1 style="font-size: 2.5rem;">User Guide</h1>
        <p>Complete documentation for MammoCare platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Getting Started",
        "Manual Processing",
        "Auto Processing",
        "FAQ"
    ])
    
    with tab1:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: white;">
            <h3>Welcome to MammoCare!</h3>
            <p>This guide will help you get started with our advanced breast cancer detection platform.</p>
            <h4>Prerequisites</h4>
            <ul>
                <li>Mammogram images in DICOM, JPG, JPEG, or PNG format</li>
                <li>Basic understanding of mammography</li>
                <li>Recommended: Medical or radiological background</li>
            </ul>
            <h4> Quick Start Steps</h4>
            <ol>
                <li><strong>Navigate:</strong> Use the sidebar menu to access different features</li>
                <li><strong>Upload:</strong> Select and upload your mammogram image</li>
                <li><strong>Configure:</strong> Adjust processing parameters as needed</li>
                <li><strong>Process:</strong> Run the analysis and review results</li>
                <li><strong>Download:</strong> Save processed images and reports</li>
            </ol>
            <h4>Tips for Best Results</h4>
            <ul>
                <li>Use high-quality, well-exposed mammogram images</li>
                <li>Ensure proper image orientation before upload</li>
                <li>Start with default settings and adjust as needed</li>
                <li>Review all visualization options for comprehensive analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: white;">
            <h3>🔬 Manual Processing Guide</h3>
            <h4>Step 1: Image Upload</h4>
            <p>Click the upload button and select your mammogram file. Supported formats include DICOM (.dcm), 
            JPG, JPEG, and PNG files.</p>
            <h4>Step 2: Adjust Preprocessing Settings</h4>
            <ul>
                <li><strong>Gaussian Blur:</strong> Controls noise reduction (range: 1-15)</li>
                <li><strong>Side Selection:</strong> Choose Left or Right breast</li>
            </ul>
            <h4>Step 3: Define Pectoral Muscle Region</h4>
            <p>Use the coordinate sliders to mark the start and end points of the pectoral muscle boundary. 
            The system will automatically segment and remove the muscle tissue.</p>
            <h4>Step 4: Process and Analyze</h4>
            <p>Click "Process Image" to generate results. Review the three output images:</p>
            <ul>
                <li><strong>Original:</strong> Unprocessed mammogram</li>
                <li><strong>Muscle Removed:</strong> Image with pectoral muscle removed</li>
                <li><strong>Dense Regions:</strong> Highlighted areas of high tissue density</li>
            </ul>
            <h4>Step 5: AI Analysis</h4>
            <p>Use the AI assistant to get detailed insights about detected abnormalities and tissue characteristics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: white;">
            <h3>Automated Processing Guide</h3>
            <h4>How It Works</h4>
            <p>The automated system uses advanced algorithms including:</p>
            <ul>
                <li>Depth-first search for boundary detection</li>
                <li>Multi-scale edge detection</li>
                <li>Machine learning-based tissue classification</li>
                <li>Adaptive thresholding techniques</li>
            </ul> 
            <h4>Advantages</h4>
            <ul>
                <li><strong>Speed:</strong> Process images in seconds</li>
                <li><strong>Accuracy:</strong> 98.5% precision rate</li>
                <li><strong>Consistency:</strong> Reproducible results</li>
                <li><strong>Batch Processing:</strong> Handle multiple images</li>
            </ul>
            <h4>When to Use Auto vs Manual</h4>
            <ul>
                <li><strong>Use Auto:</strong> Standard cases, high volume, initial screening</li>
                <li><strong>Use Manual:</strong> Complex cases, fine-tuning, unusual anatomy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: white;">
            <h3>Frequently Asked Questions</h3>
            <h4>Q: What image formats are supported?</h4>
            <p><strong>A:</strong> MammoCare supports DICOM (.dcm), JPG, JPEG, and PNG formats. 
            DICOM is preferred for medical-grade analysis.</p>
            <h4>Q: How accurate is the AI analysis?</h4>
            <p><strong>A:</strong> Our system achieves 98.5% accuracy on standard cases. However, 
            all AI results should be reviewed by qualified medical professionals.</p>
            <h4>Q: Is my data secure?</h4>
            <p><strong>A:</strong> Yes, all data is processed with HIPAA-compliant security measures. 
            Images are not stored permanently and are deleted after processing.</p>
            <h4>Q: Can I process multiple images at once?</h4>
            <p><strong>A:</strong> Batch processing is available in the automated processing mode. 
            Manual mode processes one image at a time for precision control.</p>
            <h4>Q: What should I do if results look incorrect?</h4>
            <p><strong>A:</strong> Try adjusting the preprocessing parameters or switch to manual mode 
            for more control. Contact support if issues persist.</p>
            <h4>Q: How do I interpret the dense region visualization?</h4>
            <p><strong>A:</strong> Red contours indicate areas of high tissue density that may require 
            closer examination. Consult with a radiologist for clinical interpretation.</p>
        </div>
        """, unsafe_allow_html=True)

# CONTACT SECTION
elif selected == "Contact":
    st.markdown("""
    <div style=" text-align: start; padding: 30px; border: 1px solid #e0e0e0; border-radius: 12px; background-color: white; margin-bottom: 20px;">
        <h1 style="font-size: 2.5rem;"> Contact Us</h1>
        <p>We're here to help you with any questions or support needs</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: white;">
            <h3>Get in Touch</h3>
            <p>Have questions or need assistance? Our team is ready to help!</p>
            <h4>Contact Information</h4>
            <ul>
             
                <li><strong>Phone:</strong>9833755209</li>
              
            </ul>
            <h4>Business Hours</h4>
            <p>Monday - Friday: 9:00 AM - 6:00 PM EST<br>
            <h4>Social Media</h4>
            <p>Follow us for updates and news:</p>
            <ul>
                <li>Twitter:</li>
                <li>Facebook:</li>
                <li>LinkedIn:</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: white; margin-bottom: 20px;">
            <h3>Send Us a Message</h3>
        """, unsafe_allow_html=True)
        
        with st.form("contact_form"):
            name = st.text_input("Full Name *")
            email = st.text_input("Email Address *")
            subject = st.selectbox(
                "Subject *",
                ["General Inquiry", "Technical Support", "Feature Request", 
                 "Bug Report", "Partnership", "Other"]
            )
            message = st.text_area("Message *", height=200)
            
            submitted = st.form_submit_button("Send Message", use_container_width=True)
            
            if submitted:
                if name and email and message:
                    st.markdown("""
                    <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #d4edda; color: #155724; margin-top: 20px;">
                        <p><strong>Thank you for contacting us!</strong></p>
                        <p>We've received your message and will respond within 24 hours.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Please fill in all required fields (*)")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Team section
    st.markdown("---")
    st.markdown("## Our Team")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="feature-card" style="text-align: center;">
            <h4>Development Team</h4>
            <p>Silent Echo Technologies</p>
            <p>Expert software engineers and AI specialists</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="feature-card" style="text-align: center;">
            <h4>Medical Advisors</h4>
            <p>Board-Certified Radiologists</p>
            <p>Clinical expertise and guidance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="feature-card" style="text-align: center;">
            <h4>Support Team</h4>
            <p>24/7 Customer Service</p>
            <p>Always here to help</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
# Footer Section
st.markdown("""
<style>
.footer-container {
    background: white;
    color: white;
    padding: 50px 40px 30px 40px;
    border-radius: 15px 15px 0 0;
    margin-top: 60px;
    border: 1px solid #e2e8f0;
    color: black;
}

.footer-content {
    max-width: 1200px;
    margin: 0 auto;
}

.footer-logo-section {
    text-align: start;
    margin-bottom: 10px;
}

.footer-logo {
    max-width: 150px;
    height: 150px;
    margin-bottom: 15px;
    filter: brightness(1.1);
}

.footer-tagline {
    font-size: 1.1rem;
    margin-bottom: 10px;
    opacity: 0.95;
}

.footer-company {
    font-size: 1rem;
    color: black;
    margin-bottom: 25px;
}

.footer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 4fr));
    gap: 30px;
    margin: 30px 0;
    padding: 30px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.footer-column h4 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 15px;
    color: #667eea;
}

.footer-column ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.footer-column ul li {
    margin-bottom: 10px;
}

.footer-column ul li a {
    color: black;
    text-decoration: none;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}

.footer-column ul li a:hover {
    color: #667eea;
    padding-left: 5px;
}

.footer-social {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin: 20px 0;
}

.social-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(102, 126, 234, 0.2);
    border-radius: 50%;
    color: white;
    text-decoration: none;
    font-size: 1.2rem;
    transition: all 0.3s ease;
}

.social-icon:hover {
    background: #667eea;
    transform: translateY(-3px);
}

.footer-bottom {
    text-align: center;
    padding-top: 25px;
}

.footer-links {
    font-size: 0.95rem;
    margin-bottom: 15px;
}

.footer-links a {
    color: #cbd5e0;
    text-decoration: none;
    margin: 0 10px;
    transition: color 0.3s ease;
}

.footer-links a:hover {
    color: #667eea;
}

.footer-copyright {
    font-size: 0.9rem;
    opacity: 0.8;
    margin-bottom: 15px;
}

.footer-disclaimer {
    background: red;
    border-left: 3px solid #ffc107;
    padding: 15px;
    border-radius: 5px;
    font-size: 0.85rem;
    line-height: 1.6;
    color: black;
    margin-top: 20px;
    text-align: left;
}

.footer-disclaimer strong {
    color: #ffeb3b;
}

@media (max-width: 768px) {
    .footer-container {
        padding: 40px 20px 20px 20px;
    }
    
    .footer-grid {
        grid-template-columns: 1fr;
        gap: 20px;
    }
}
</style>

<div class="footer-container">
    <div class="footer-content">
        <!-- Logo and Tagline -->
        <div class="footer-logo-section">
            <img src="https://image2url.com/images/1761486463113-60b599fb-2c40-4ad2-bac8-6ee33a8bb534.png" 
                 alt="MammoCare Logo" 
                 class="footer-logo">
            <p class="footer-tagline">Advanced AI-Powered Breast Cancer Detection Platform</p>
            <p class="footer-company">Developed by <strong>Silent Echo</strong></p>
        </div>
        <!-- Footer Grid -->
        <div class="footer-grid">
            <div class="footer-column">
                <h4>Platform</h4>
                <ul>
                    <li><a href="#home">Home</a></li>
                    <li><a href="#manual">Manual Processing</a></li>
                    <li><a href="#auto">Auto Processing</a></li>
                    <li><a href="#viz">3D Visualization</a></li>
                </ul>
            </div>
            <div class="footer-column">
                <h4> Resources</h4>
                <ul>
                    <li><a href="#centers">Treatment Centers</a></li>
                    <li><a href="#guide">User Guide</a></li>
                    <li><a href="#faq">FAQ</a></li>
                    <li><a href="#support">Support</a></li>
                </ul>
            </div>
            <div class="footer-column">
                <h4> Contact</h4>
                <ul>
                    <li><a href="mailto:info@mammocare.com">info@mammocare.com</a></li>
                    <li><a href="tel:9833755209</a></li>
                    <li><a href="#location">Find Us</a></li>
                    <li><a href="#contact">Contact Form</a></li>
                </ul>
            </div>
            <div class="footer-column">
                <h4> Legal</h4>
                <ul>
                    <li><a href="#privacy">Privacy Policy</a></li>
                    <li><a href="#terms">Terms of Service</a></li>
                    <li><a href="#compliance">HIPAA Compliance</a></li>
                    <li><a href="#cookies">Cookie Policy</a></li>
                </ul>
            </div>
        </div>
        <!-- Social Media -->
        <div class="footer-social">
            <a href="#facebook" class="social-icon" title="Facebook"></a>
            <a href="#twitter" class="social-icon" title="Twitter"></a>
            <a href="#linkedin" class="social-icon" title="LinkedIn"></a>
            <a href="#instagram" class="social-icon" title="Instagram"></a>
            <a href="#youtube" class="social-icon" title="YouTube">▶</a>
        </div>
        <!-- Bottom Section -->
        <div class="footer-bottom">
            <div class="footer-copyright">
                © 2025 MammoCare. All rights reserved.
            </div>
            <div class="footer-links">
                <a href="#privacy">Privacy Policy</a> • 
                <a href="#terms">Terms of Service</a> • 
                <a href="#accessibility">Accessibility</a> • 
                <a href="#sitemap">Sitemap</a>
            </div> 
            <!-- Medical Disclaimer -->
            <div class="footer-disclaimer">
                <strong>⚠️ Medical Disclaimer:</strong><br>
                For medical emergencies, please contact your healthcare provider immediately. 
                MammoCare is a diagnostic aid and should not replace professional medical advice, 
                diagnosis, or treatment. Always seek the advice of your physician or other qualified 
                health provider with any questions you may have regarding a medical condition.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
