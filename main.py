import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from googletrans import Translator

# Load Model Function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize image
    
    predictions = model.predict(input_arr)
    predicted_class = np.argmax(predictions)  # Get the class with the highest probability
    
    return predicted_class, predictions[0]  # Return full probability array

# Class Labels
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease Solutions
disease_tips = {
    'Apple___Apple_scab': "Use fungicides and remove infected leaves.",
    'Apple___Black_rot': "Prune infected branches and apply copper sprays.",
    'Tomato___Late_blight': "Use resistant varieties and apply fungicides.",
    'Tomato___Tomato_mosaic_virus': "Destroy infected plants and control insects.",
    'Potato___Early_blight': "Use certified disease-free seeds and crop rotation.",
    'Grape___Black_rot': "Remove affected grapes and use fungicides.",
    'Corn_(maize)___Northern_Leaf_Blight': "Remove affected grapes and use fungicides."
}

# Multi-Language Support
translator = Translator()
languages = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te"}
selected_lang = st.sidebar.selectbox("🌍 Select Language", list(languages.keys()))

def translate_text(text):
    return translator.translate(text, dest=languages[selected_lang]).text if selected_lang != "English" else text

# Sidebar
st.sidebar.title(translate_text("Plant Disease Detection System"))
app_mode = st.sidebar.selectbox(translate_text("Select Page"), ["HOME", "DISEASE RECOGNITION"])

# Display an Image
img = Image.open("Diseases.png")
st.image(img, use_container_width=True)

# Home Page
if app_mode == "HOME":
    st.markdown(f"<h1 style='text-align: center;'>{translate_text('Plant Disease Detection System for Sustainable Agriculture')}</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header(translate_text("Plant Disease Detection System for Sustainable Agriculture"))
    
    test_image = st.file_uploader(translate_text("Choose an Image:"))
    
    if test_image:
        st.image(test_image, use_container_width=True)

        if st.button(translate_text("Predict")):
            st.write(translate_text("Our Prediction"))
            
            result_index, probabilities = model_prediction(test_image)

            predicted_disease = class_name[result_index]
            st.success(f"{translate_text('Model Prediction')}: **{translate_text(predicted_disease)}**")

            # Probability Bar Chart
            st.bar_chart(probabilities)

            # Show Treatment Advice
            if predicted_disease in disease_tips:
                st.warning("⚠️ " + translate_text("Treatment & Prevention:") + " " + translate_text(disease_tips[predicted_disease]))
            else:
                st.info(translate_text("No specific treatment found. Consult an expert."))
