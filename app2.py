import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

#  Full bilingual crop information (22 crops)
crop_info = {
    "Rice": {
        "en": "Grows well in clayey soil and needs a lot of water.",
        "ta": "களிமண் மண்ணில் நன்றாக வளர்ந்து, அதிகமான நீர் தேவை."
    },
    "Maize": {
        "en": "Prefers well-drained fertile soil with moderate rainfall.",
        "ta": "சராசரி மழை கொண்ட, நல்ல நீர் வடிகால் மண் விரும்புகிறது."
    },
    "Chickpea": {
        "en": "Thrives in loamy soil with low humidity.",
        "ta": "சிறிய ஈரப்பதம் கொண்ட லோமி மண்ணில் நன்றாக வளரும்."
    },
    "Kidneybeans": {
        "en": "Requires warm climates and well-drained soil.",
        "ta": "வெப்பமான காலநிலையும், நன்கு வடிகாலான மண்ணும் தேவை."
    },
    "Pigeonpeas": {
        "en": "Grows in semi-arid conditions with light soil.",
        "ta": "அரிதான மழை வானிலையிலும், இலகுவான மண்ணிலும் வளரும்."
    },
    "Mothbeans": {
        "en": "Tolerates drought, grows in sandy soil.",
        "ta": "வறட்சியை தாங்கும் தன்மை கொண்டது, மணல் மண்ணில் வளரும்."
    },
    "Mungbean": {
        "en": "Requires hot, dry weather with moderate water.",
        "ta": "வெப்பமான மற்றும் உலர்ந்த வானிலை தேவை, சிறிய நீர்த் தேவையுடன்."
    },
    "Blackgram": {
        "en": "Grows in warm and humid conditions.",
        "ta": "வெப்பமான மற்றும் ஈரப்பதம் நிறைந்த சூழலில் வளரும்."
    },
    "Lentil": {
        "en": "Needs cool weather and fertile loamy soil.",
        "ta": "குளிரான வானிலை மற்றும் வளமான லோமி மண் தேவை."
    },
    "Pomegranate": {
        "en": "Prefers well-drained sandy loam soil in dry climate.",
        "ta": "உலர்ந்த காலநிலைக்கு ஏற்ப, மணல்வெட்டி மண்ணை விரும்புகிறது."
    },
    "Banana": {
        "en": "Requires rich loamy soil and high temperature.",
        "ta": "சரிவான லோமி மண் மற்றும் உயர் வெப்பநிலை தேவை."
    },
    "Mango": {
        "en": "Requires warm and dry climate with sandy loam soil.",
        "ta": "வெப்பமான மற்றும் உலர்ந்த காலநிலை மற்றும் மணல்வெட்டி மண் வேண்டும்."
    },
    "Grapes": {
        "en": "Needs hot and dry climate, well-drained soil.",
        "ta": "கடுமையான வெப்பமும் உலர்ந்த காலநிலையும், நன்கு வடிகாலான மண் தேவை."
    },
    "Watermelon": {
        "en": "Needs sandy soil and hot climate with good irrigation.",
        "ta": "மணல் மண் மற்றும் நன்கு பாசன வசதி கொண்ட வெப்பமான வானிலை தேவை."
    },
    "Muskmelon": {
        "en": "Prefers dry and warm climate, well-drained sandy soil.",
        "ta": "உலர்ந்த மற்றும் வெப்பமான காலநிலை மற்றும் நன்கு வடிகாலான மணல் மண்."
    },
    "Apple": {
        "en": "Requires cold climate and fertile soil.",
        "ta": "குளிர்ந்த வானிலை மற்றும் வளமான மண் தேவை."
    },
    "Orange": {
        "en": "Grows in subtropical climate and well-drained soil.",
        "ta": "அரைப் பருவவானிலை மற்றும் நன்கு வடிகாலான மண்ணில் வளரும்."
    },
    "Papaya": {
        "en": "Thrives in warm climate with rich soil and irrigation.",
        "ta": "வெப்பமான காலநிலை மற்றும் பாசன வசதி உள்ள வளமான மண்ணில் வளரும்."
    },
    "Coconut": {
        "en": "Needs sandy loam soil and coastal humid climate.",
        "ta": "மணல்வெட்டி மண் மற்றும் கடற்கரை ஈரமான காலநிலை தேவை."
    },
    "Cotton": {
        "en": "Prefers black soil and warm climate.",
        "ta": "கரிமண் மற்றும் வெப்பமான காலநிலை விருப்பம்."
    },
    "Jute": {
        "en": "Grows in warm, humid areas with loamy soil.",
        "ta": "வெப்பமான மற்றும் ஈரமான வானிலை கொண்ட லோமி மண்ணில் வளரும்."
    },
    "Coffee": {
        "en": "Requires shaded, humid conditions with rich soil.",
        "ta": "நிழலுடன் ஈரப்பதமான சூழ்நிலை மற்றும் வளமான மண் தேவை."
    }
}

# App Layout
st.set_page_config(page_title="Smart Crop Recommender 🌱", layout="wide")
st.title("🌾 Smart Crop Recommendation System")

# Sidebar Inputs
st.sidebar.header("📋 Enter Soil & Weather Details")

N = st.sidebar.number_input(" Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.sidebar.number_input(" Phosphorus (P)", min_value=5, max_value=145, value=42)
K = st.sidebar.number_input(" Potassium (K)", min_value=5, max_value=205, value=43)
temperature = st.sidebar.number_input(" Temperature (°C)", min_value=8.0, max_value=45.0, value=20.5)
humidity = st.sidebar.number_input(" Humidity (%)", min_value=10.0, max_value=100.0, value=80.0)
ph = st.sidebar.number_input(" Soil pH", min_value=3.5, max_value=9.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=200.0)

# Feature Engineering
K_log = np.log1p(K)
humidity_sq = humidity ** 2
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, K_log, humidity_sq]])

# Prediction Logic
if st.sidebar.button("Recommend Crop"):
    prediction = model.predict(input_data)
    crop = prediction[0].title()

    st.markdown(f"Recommended Crop: **{crop}**")

    with st.expander("Crop Details"):
        st.write("**English:**", crop_info.get(crop, {}).get("en", "No info available."))
        st.write("**தமிழ்:**", crop_info.get(crop, {}).get("ta", "தகவல் இல்லை."))

        # Show Crop Image if available
        image_path = os.path.join("crop_images", f"{crop.lower()}.jpg")
        if os.path.exists(image_path):
            img = Image.open(image_path)
            st.image(img, caption=f"{crop} Photo", use_container_width=True)
        else:
            st.warning(" Crop photo not available.")
else:
    st.info("Please enter the soil & weather details and click **Recommend Crop**.")
