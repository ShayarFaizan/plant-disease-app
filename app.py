import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("plant_disease_model.keras")

class_names = list(train_data.class_indices.keys())

st.title("Plant Disease Classifier 🌿")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)

    if confidence < 0.70:
        st.write("Prediction: Unknown")
    else:
        st.write("Prediction:", class_names[predicted_class])
        st.write("Confidence:", round(confidence*100,2), "%")

