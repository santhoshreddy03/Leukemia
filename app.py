import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('leukemia_cnn_model.h5')

# Function to predict the image
def predict_image(img, model):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction < 0.5:
        return "Normal"
    else:
        return "Cancer"

# Streamlit app
st.title("Leukemia Cancer Detection")
st.write("Upload an image to predict whether it is of a cancerous cell or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Predict the image
    prediction = predict_image(img, model)
    if prediction == "Normal":
        st.success(f"The image is classified as: **{prediction}**")
    else:
        st.error(f"The image is classified as: **{prediction}**")

# Disclaimer at the bottom
st.write("""
    ---
    **Disclaimer: This app is for educational purposes only and should not be used for medical diagnosis.**
""")

# Attracting features
st.sidebar.title("About the App")
st.sidebar.info("""
    This app uses a Convolutional Neural Network (CNN) model to classify leukemia cancer images.
    Upload an image of a cell to see if it is cancerous or normal.
    The model was trained using TensorFlow and Keras.
""")

st.sidebar.title("How it Works")
st.sidebar.info("""
    1. Upload an image of a cell.
    2. The image is processed and passed through the CNN model.
    3. The model predicts whether the cell is cancerous or normal.
    4. The prediction result is displayed on the screen.
""")

st.sidebar.title("Contact")
st.sidebar.info("""
    If you have any questions or feedback, please contact:
    - Email: padalasantosh.reddy2021@vitstudent.ac.in
    - GitHub: [santhoshreddy03] (https://github.com/santhoshreddy03)
""")

