import streamlit as st
from PIL import Image
import requests

# Set page configuration
st.set_page_config(page_title="Myoptic AI", page_icon="🤖", layout="centered")

# Load and display the logo
logo_path = "MyopticAI.png"  # Ensure the file is in the same directory or provide the correct path
st.image(logo_path, use_container_width=True)

# Custom styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Poppins:wght@300;600&display=swap');
        body {
            font-family: 'Poppins', sans-serif;
        }
        .purple-box {
            background-color: #4B0082;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 32px;
            font-weight: 700;
            font-family: 'Montserrat', sans-serif;
        }
        .purple-box span {
            font-size: 18px;
            font-weight: 300;
        }
        .white-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: left;
            color: #4B0082;
            font-size: 18px;
            font-family: 'Poppins', sans-serif;
            border: 2px solid #4B0082;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display Welcome Message in a Purple Box
st.markdown('<div class="purple-box">WELCOME TO MYOPTIC AI!<br><span>We are here to take care of you!</span></div>', unsafe_allow_html=True)

# About Us Section in a White Box
about_text = """
### About Us

Our team consisting of Mahi, Yuvika, Manan, and Rishika, have developed an AI-integrated project titled *'MyOptic AI'*, which is an *Artificial Intelligence-driven Multi-Model Framework for Vision Care* that includes a prediction and recommendation system that keeps a record of a patient's eye health based on various parameters.

This project integrates deep learning, machine learning, and natural language processing to provide a comprehensive AI-driven Vision Care system. The CNN-based analysis enables accurate detection of retinal abnormalities through image processing, while the RFC-based prediction model offers data-driven insights based on numerical parameters. Additionally, the LLM-based chatbot enhances user engagement by providing personalized recommendations. By combining these three approaches, our system aims to assist individuals in monitoring their eye health efficiently, promoting early detection, and encouraging preventive care.
"""

st.markdown(f'<div class="white-box">{about_text}</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Enter Data Manually", "Upload Retinal Scans", "Chatbot"])

RFC_API_URL = "http://127.0.0.1:8000/predict_rfc"
# Enter Data Manually Page
if page == "Enter Data Manually":
    st.title("Enter Data Manually")
    
    # User Inputs
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    spherical_equivalence = st.number_input("Spherical Equivalence")
    axial_length = st.number_input("Axial Length (mm)")
    anterior_chamber_depth = st.number_input("Anterior Chamber Depth (mm)")
    lens_thickness = st.number_input("Lens Thickness (mm)")
    vitreous_chamber_depth = st.number_input("Vitreous Chamber Depth (mm)")
    weekly_sports_hours = st.number_input("Weekly hrs spent playing sports")
    weekly_reading_hours = st.number_input("Weekly hrs spent reading")
    weekly_computer_hours = st.number_input("Weekly hrs spent on computer")
    weekly_study_hours = st.number_input("Weekly hrs spent studying")
    weekly_tv_hours = st.number_input("Weekly hrs spent watching TV")
    diopter_hours = st.number_input("Diopter hrs (weekly work-load)")
    father_myopic = st.selectbox("Is your father myopic?", ["Yes", "No"])
    mother_myopic = st.selectbox("Is your mother myopic?", ["Yes", "No"])

    # Button to get results
    if st.button("Get Results"):
        user_data = {
            "GENDER": gender,
            "AL": axial_length,
            "ACD": anterior_chamber_depth,
            "LT": lens_thickness,
            "VCD": vitreous_chamber_depth,
            "SPHEQ": spherical_equivalence
        }

        # Send data to FastAPI backend
        response = requests.post(RFC_API_URL, json=user_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
            st.info(f"Confidence: **{result['confidence']:.2f}%**")
        else:
            st.error("Error occurred while fetching results.")

# Upload Retinal Scans Page
elif page == "Upload Retinal Scans":
    st.title("Upload Retinal Scans and Images")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Retinal Scan", use_container_width=True)
        
        # Send image to FastAPI
        with st.spinner("Analyzing image..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/predict_cnn", files=files)

            if response.status_code == 200:
                st.success("Image processed successfully!")
                st.json(response.json())  # Display response from FastAPI
            else:
                st.error("Failed to process the image.")

# Chatbot Page
elif page == "Chatbot":
    st.title("Myoptic AI Chatbot")
    st.write("Hi, This is Myoptic AI chatbot. What issue are you facing?")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

    # User input
    user_input = st.text_input("Type your message here:")
    
    if st.button("Send") and user_input.strip():
        try:
            # Send request to FastAPI chatbot
            api_url = "http://127.0.0.1:8000/chat"  # Ensure FastAPI server is running
            response = requests.post(api_url, json={"message": user_input})

            if response.status_code == 200:
                bot_reply = response.json().get("reply", "No response received.")
            else:
                bot_reply = "Error: Unable to connect to chatbot."

            # Save conversation history
            st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})

            # Refresh chat display
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
