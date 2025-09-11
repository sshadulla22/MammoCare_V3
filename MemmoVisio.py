import streamlit as st
import pandas as pd
import numpy as np
import snowflake.connector
import streamlit_option_menu
from streamlit_option_menu import option_menu
import os
from PIL import Image
import pydicom  # Import pydicom for DICOM file support
from streamlit_chat import message  # Import the message component for chat
import requests  # To make HTTP requests to aiXplain API
import time  # For polling delay
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_multiotsu
import streamlit.components.v1 as components
import io



# Set the API key for aiXplain
API_KEY = '799b8640ed5d2e45959f34bc3adf4f4c45515d0d492d171b8e7f07cd0da48c1e'
API_URL = 'https://models.aixplain.com/api/v1/execute/64788e666eb56313aa7ebac1'  # aiXplain model URL

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Set page configuration
st.set_page_config(page_title="MammoCare", page_icon="🩺", layout="wide")

#Load CSV DATA----Shadulla Shaikh Date of Update 02-11-2024 Time: 18:40
@st.cache_data
def load_data():
    try:
        return pd.read_csv('treatment_centers.csv', encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv('treatment_centers.csv', encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None  # Return None if there's an error

# Load the data from CSV fible Updated by Shadulla Shaikh 02-11-2024 Time: 18:42
data = load_data()

# Inject custom styles to modify the default Streamlit header and make the button responsive
st.markdown(
    """
    <style>
        /* Custom header styling */
        .stAppHeader {
            background-color: #31333f !important;
            color: white !important;
            padding: 10px 20px !important;
            border-bottom: 2px solid #ccc !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            z-index: 1000 !important;
        }

        .stAppToolbar {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
        }

        /* Add a button to the header 
        .custom-header-button {
            background-color: #007BFF !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            font-size: 14px !important;
            cursor: pointer !important;
            border-radius: 5px !important;
            text-decoration: none !important;
            margin-right: 10px !important;
            display: inline-block;
        }

        .custom-header-button:hover {
            background-color: #0056b3 !important;
        }
        */
        /* Prevent overlap with fixed header */
        .main-content {
            margin-top: 70px;
        }

        /* Make button responsive 
        @media (max-width: 768px) {
            .custom-header-button {
                font-size: 12px !important;
                padding: 8px 16px !important;
            } */

            /* Adjust header layout for small screens */
            .stAppToolbar {
                flex-direction: column !important;
                align-items: center !important;
            }

            .stAppHeader {
                padding: 1px 1px !important;
            }

              /* Three-dot menu styling */
        .stMainMenu {
            color: white !important;
        }

            /* Adjust text in header */
            .stAppHeader h1 {
                font-size: 40px !important;
                align-items: center !important;
            }
        }

        @media (max-width: 480px) {
            .custom-header-button {
                font-size: 10px !important;
                padding: 6px 12px !important;
            }

            /* Adjust header layout for very small screens */
            .stAppToolbar {
                flex-direction: column !important;
                align-items: center !important;
            }

            .stAppHeader {
                padding: 1px 1px !important;
            }

            /* Adjust text in header for small screens */
            .stAppHeader h1 {
                font-size: 40px !important;
                text-align: center !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Add a custom header with a button
st.markdown(
    """
    <header class="stAppHeader">
        <div class="stAppToolbar">
            <div>
                <h1 style="margin: 0; font-size: 20px; color: white;">MammoCare</h1>
            </div>
        <!-- <div>
    <a href="https://your-link-here.com" target="_blank" class="custom-header-button">Contact Us</a>
</div> -->
        </div>
    </header>
    """,
    unsafe_allow_html=True
)



# Add CSS styles for better formatting


st.markdown(
    """
    <style>

      
    body {
        background-color: #f4f4f4;
    }
    .hero {
        background-image: url('https://cdn.apollohospitals.com/health-library-prod/2021/04/3D-MAMMOGRAM-scaled.jpg');
        background-size: cover;
        background-position: center;
        height: 400px;
        border-radius: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        border: 5px solid #31333F;
        
    }
    .cta-button {
        background-color: Black;
        padding: 20px 30px;
        color: #31333F;
        border-radius: 30px;
        text-decoration: none;
        font-weight: bold;
        border:5px solid #31333F;
        box-shadow: 0 0 10px rgba(0.1, 0.1, 0.1, 0.9);
    }
    .cta-button:hover {
        background-color: white;
    }
    .section {
        padding: 50px;
     text-align: justify;
    }
    .feature-box {
        background-color: white;
        padding: 20px;
        color:black;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }


  
    
    </style>
    """,
    unsafe_allow_html=True
)



with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "MP Muscle Removal", "AP Muscle Removal", "Treatment Centers", "How to Use This Software", "Presentation Deck", "Contact Us"],
        icons=["house", "activity", "activity", "book", "globe", "play", "cogs"],
        menu_icon="cast",
        default_index=0
    )

    # Adding custom text below the sidebar menu
    
    st.markdown("<p style='text-align: center; margin-top: -12px;'>AI-Powered Diagnostic Assistance</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-top: -22px;'>Developed by Silent Echo</p>", unsafe_allow_html=True)
     
    
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
    else:  # Right
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
    data = {
        'text': user_input
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 201:
            result = response.json()
            request_id = result.get('requestId')
            if request_id:
                return poll_aixplain_model(request_id)  
        else:
            return f"Error: API request failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception occurred: {e}"
    
    # Example NumPy array representing your image
# Replace this with your actual processed NumPy array
# For demonstration, creating a dummy NumPy array
muscle_removed_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)





#----------------------------------------------------------------------------------------------------------------------

# Home Section
if selected == "Home":
       # Hero section
    st.markdown("""
    <style>
        .hero {
            text-align: center;
            padding: 20px;
        }

        .hero h1 {
            font-size: 40px;
            color: white;
            text-align: center;
            padding: 20px;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3), 0 0 20px #495361, 0 0 5px darkblue;
        }

        /* Responsive styles */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 30px; /* Reduce font size for smaller screens */
                padding: 15px;
            }
        }

        @media (max-width: 480px) {
            .hero h1 {
                font-size: 30px; /* Further reduce font size for very small screens */
                padding: 10px;
            }
        }
    </style>

    <div class="hero">
        <h1>
            "Enhancing Breast Cancer Detection by Eliminating Unwanted Artifacts like Pectoral Muscle"
        </h1>
        
    </div>
""", unsafe_allow_html=True)

    # st.markdown("<h2 style='text-align:center; margin-top: -12px;'>Advanced Mammogram Image Processing</h2>", unsafe_allow_html=True)
   


   # About Section with Responsive CSS
    st.markdown("""
    <style>
        /* Style for the About section */
        #About-box {
            margin-top: 20px;
            margin-bottom: 30px;
        }
        #About-box h1 {
            font-size: 40px;
            color: #495361;
            margin-bottom: 12px;    
            text-align: center;
        }
        #About-box p {
            font-size: 20px;
            line-height: 1.6;
            text-align: justify;
            color: #333;
            padding: 10px;
        }

        /* Media Queries for Responsiveness */
        @media (max-width: 768px) {
            #About-box h1 {
                font-size: 30px;
            }
            #About-box p {
                font-size: 16px;
                padding: 5px;
            }
        }

        @media (max-width: 480px) {
            #About-box h1 {
                font-size: 30px;
            }
            #About-box p {
                font-size: 16px;
                padding: 5px;
            }
        }
    </style>

    <div class="About-box" id="About-box">
        <h1>About MammoCare</h1>
        <p>
            MammoCare is a cutting-edge mammogram image processing platform designed to enhance the clarity and quality of mammogram images through advanced preprocessing techniques.
            The platform is dedicated to optimizing image clarity, identifying dense regions, and providing healthcare professionals with the detailed visual information necessary for accurate assessments.
            MammoCare employs manual techniques for pectoral muscle removal, utilizing adjustment methods that the user manually defines to eliminate the pectoral muscle based on the adjustments.
            The Auto Pectoral Muscle Removal technique, including depth-first search algorithms and various image processing methods, achieves efficient and precise muscle segmentation and removal.
            The integration of these methodologies ensures optimal visualization of breast tissue, leading to enhanced diagnostic reliability. 
            By focusing on high-level image preprocessing, MammoCare empowers healthcare professionals to make informed decisions, ultimately improving patient care in breast health management. 
            MammoCare also integrates aiXplain AI-Powered Diagnostic Assistance to further enhance diagnostic accuracy and clinical decision-making.
        </p>
    </div>
""", unsafe_allow_html=True)

    # Features section
    st.markdown("<div class='About-box' id='About-box'><h1>Key Features</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3 class='feature-box'>Advanced Image Clarity Optimization</h3><p>MammoCare utilizes high-level preprocessing techniques to enhance the clarity and quality of mammogram images. This allows for better visualization of breast tissue, enabling healthcare professionals to make more accurate assessments.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3 class='feature-box'>Dual Approach to Pectoral Muscle Removal</h3><p> MammoCare employs manual techniques for pectoral muscle removal, utilizing adjustment methods that user manually define and eliminate the pectoral muscle based on the adjustments. The Auto Pectoral Muscle Removal technique, including depth-first search algorithms and various image processing methods, to achieve efficient and precise muscle segmentation and removal.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3 class='feature-box'>Dense Region Identification</h3><p>MammoCare focuses on identifying dense regions within mammogram images. By highlighting these areas, healthcare professionals can quickly and effectively assess potential abnormalities, facilitating timely and informed decision-making in breast health management.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3 class='feature-box'>AI-Powered Diagnostic Assistance</h3><p>Integrated with aiXplain, MammoCare offers AI-driven diagnostic assistance, providing healthcare professionals with valuable insights and recommendations based on advanced algorithms. This feature enhances diagnostic accuracy and supports clinicians in making well-informed decisions regarding patient care.</p></div>", unsafe_allow_html=True)

    # # Sidebar for user input
    # st.sidebar.header("AI Model Interaction")
    # user_input = st.sidebar.text_input("Enter diagnosis request or question for the AI model", key="user_input")  # Unique key provided
    # if st.sidebar.button("Ask"):
    #     if user_input.strip():  # Check if the input is not empty
    #         with st.spinner("Processing your request..."):
    #             response = query_aixplain_model(user_input)
    #         st.sidebar.success(response)
    #     else:
    #         st.sidebar.error("Please enter a valid request or question.")



    # # Footer
    # st.markdown("<footer style='text-align: center; padding: 20px; background-color: #E91E63; color: white;'>© 2024 MammoCare. All rights reserved.</footer>", unsafe_allow_html=True)
   
        
    
if selected == "MP Muscle Removal":
    st.title("Manual Pectoral Muscle Removal & Dense Regoin Visualization")
    # st.title("Manual Pectoral Muscle Removal & Dense Regoin Visualization")
    # Show Instructions at the top left and expand by default
    with st.expander("Instructions", expanded=True):
     st.markdown("""
    ### How to Use the Application:
    
    Follow these steps to upload and process mammogram images:

    1. **Upload Image**:
        - Upload a mammogram image using the uploader.
        - Supported formats: PNG, JPG, JPEG, and DICOM (.dcm).

    2. **Adjust Settings**:
        - **Apply Equalization**: Enhance the image contrast to improve visibility.
        - **Blur Kernel Size**: Control the level of Gaussian blurring to reduce noise and enhance features.
        - **Side Selection**: Specify the side of the breast for processing.
        - **Start & End Coordinates**: Define the region for pectoral muscle removal, which will assist in image clarity.

    3. **Display Settings**:
        - Choose the processed images to display, including:
            - **Original Image**.
            - **Preprocessed Image**.
            - **Muscle Removed Image**.
            - **High Density Region Detection**.

    4. **Chatbot**:
        - Use the chatbot to ask questions or get further information about the mammogram analysis process.

    5. **Download Processed Image**:
        - After processing, download the muscle removed image or any other processed images for further review.

    ### Tips:
    - Make sure the uploaded image has clear visibility of the breast tissue for optimal results.
    - Adjust the blur kernel size and coordinates based on the image quality for better processing results.
    """, unsafe_allow_html=True)
    
    # Upload DICOM or Image File
    uploaded_file = st.file_uploader("Upload a DICOM file or a mammogram image", type=["dcm", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'):
            image = load_dicom(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Display uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocessing options
        st.sidebar.header("Image Preprocessing Options")
        blur_kernel_size = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 5, 2)

        # Preprocess the image
        preprocessed_image = preprocess_image(image, blur_kernel_size)
        with col2:
            st.image(preprocessed_image, caption="Preprocessed Image", use_container_width=True)

        # Pectoral muscle removal options
        st.sidebar.header("Pectoral Muscle Removal")
        side = st.sidebar.selectbox("Select Side", ["Left", "Right"])
        start_point = st.sidebar.slider("Start Point (X, Y)", 0, image.shape[1]-1, (0, 0), 1)
        end_point = st.sidebar.slider("End Point (X, Y)", 0, image.shape[1]-1, (image.shape[1]-1, image.shape[0]-1), 1)

        # Remove pectoral muscle from image
        muscle_removed_image = remove_pectoral_muscle(preprocessed_image.copy(), side, start_point, end_point)

        # Show muscle removed image side by side with the original
        col1, col2,col3 = st.columns(3)
        with col1:
            st.image(muscle_removed_image, caption="Muscle Removed Image", use_container_width=True)
             # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(muscle_removed_image)

    # Display the image
            st.image(pil_image, caption="Muscle Removed Image", use_container_width=True)

    # Convert the PIL Image to a downloadable format (e.g., JPEG)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            buffer.seek(0)

    # Add the download button
            st.download_button(
        label="Download Image",
        data=buffer,
        file_name="muscle_removed_image.jpg",
        mime="image/jpeg"
    )
        # High-density region detection
        thresholded_image, highest_dense_image = find_highest_dense_region(muscle_removed_image)
        
        with col2:
            st.image(thresholded_image, caption="High Density Region", use_container_width=True)

        # Show highest dense region image side by side
        
        with col3:
            st.image(highest_dense_image, caption="Highest Dense Region", use_container_width=True)

        # User input for AI model
user_input = st.text_input("Enter your diagnosis request or question for the AI model")
if st.button("Submit"):
    if user_input.strip():  # Check if the input is not empty
        with st.spinner("Processing your request..."):
            response = query_aixplain_model(user_input)
        st.success(response)
    else:
        st.error("Please enter a valid request or question.")
   

    
if selected == "AP Muscle Removal":
    # st.subheader(f"**You Have selected {selected}**")
    st.title("Automated Pectoral Muscle Removal")
    st.markdown("""
    <p>The Auto Pectoral Muscle Removal technique, including depth-first search algorithms and various image processing methods, to achieve efficient and precise muscle segmentation and removal.</p>
    <br><br><p><a href="https://bcdauto.streamlit.app/" class="cta-button">Auto Pectoral Muscle Removal</a></p><br><br>
    """, unsafe_allow_html=True)


if selected == "Treatment Centers":
    st.title("Search for treatment centers based on country, center name, or town.")
    
    # Sidebar for search filters
    st.sidebar.header("Filter Options")
    country_filter = st.sidebar.text_input("Search by Country", "")
    centre_filter = st.sidebar.text_input("Search by Center Name", "")
    town_filter = st.sidebar.text_input("Search by Town", "")

    # Initialize a boolean mask for filtering
    mask = pd.Series([True] * len(data))  # Start with all True values

    # Apply filters
    if country_filter:
        mask &= data['Country'].str.contains(country_filter, case=False, na=False)
    if centre_filter:
        mask &= data['Centre'].str.contains(centre_filter, case=False, na=False)
    if town_filter:
        mask &= data['Town'].str.contains(town_filter, case=False, na=False)

    # Filtered data
    filtered_data = data[mask]

    # Display results in the main area
    st.subheader("Search Results")
    if not filtered_data.empty:
        st.write(f"Found {len(filtered_data)} center(s):")
        st.dataframe(filtered_data)  # Display results in a table format
    else:
        st.warning("No centers found. Please refine your search.")

    # Aggregation for graph
    country_counts = data['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Number of Centers']

    # Bar chart for countries with highest treatment centers
    st.subheader("Number of Treatment Centers by Country")
    st.bar_chart(country_counts.set_index('Country'))

    # Aggregation for towns
    town_counts = data['Town'].value_counts().reset_index()
    town_counts.columns = ['Town', 'Number of Centers']

    # Bar chart for towns with highest treatment centers
    st.subheader("Number of Treatment Centers by Town")
    st.bar_chart(town_counts.set_index('Town'))





if selected == "How to Use This Software":
    st.title("How to Use This Software")
    
    # Custom CSS for styling the video container
    st.markdown("""
        <style>
            .video-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 80vh;  /* Adjust height to fit screen */
                background-color: #f9f9f9;  /* Light background */
                border: 1px solid #e0e0e0; /* Border around video */
                border-radius: 8px; /* Rounded corners */
                padding: 20px;  /* Padding around video */
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.write("Watch the tutorial video below to learn how to use this software effectively.")

    # YouTube video embedding with autoplay and mute enabled
    youtube_video_id = "9SE6B0h-4-Q"  # Replace with your actual YouTube video ID
    video_file_path = f"https://www.youtube.com/embed/{youtube_video_id}?autoplay=1&mute=1"

    # Embed video in a full-width container
    st.markdown(f"""
        <div class="video-container">
            <iframe width="100%" height="100%" src="{video_file_path}" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("For more information, visit our documentation.")



if selected == "Presentation Deck":
    st.title('Presentation Deck')

    # Embed the Canva design using iframe
    canva_iframe_code = """
    <div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
    padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
    border-radius: 8px; will-change: transform;">
    <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https://www.canva.com/design/DAGSuE39DdE/S6k_tKSQB4iHaG7GMzL4_Q/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
    </iframe>
    </div>
    <a href="https://www.canva.com/design/DAGSuE39DdE/S6k_tKSQB4iHaG7GMzL4_Q/view?utm_content=DAGSuE39DdE&utm_campaign=designshare&utm_medium=embeds&utm_source=link" target="_blank" rel="noopener">MIAS DDSM InBreast</a> by MOHDSHADULLA SHAIKH
    """
    st.markdown(canva_iframe_code, unsafe_allow_html=True)    
    


if selected == "Contact Us":  
    st.title("Contact Us")
    st.markdown("""
    <p>If you have any questions or need support, please reach out to us at <a href="mailto:support@mammocare.com">support@mammocare.com</a>.</p>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<footer style='text-align: center; padding: 20px; background-color:Black; color: white;'>© 2024 MammoCare. All rights reserved.</footer>", unsafe_allow_html=True)       


