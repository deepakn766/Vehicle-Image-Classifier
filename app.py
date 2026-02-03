import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle  # for loading trained model

# --- Function to preprocess image into DataFrame row ---
def preprocess_image(image_file, size=(64,64)):
    # Read image from uploaded file
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize and flatten
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image_np = np.array(image, dtype=np.uint8).flatten()

    # Convert to DataFrame (single row)
    df = pd.DataFrame([image_np])
    return df

# --- Streamlit UI ---
st.title("Vehicle Image Classifier 🚗✈️🚲")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess → DataFrame
    df_input = preprocess_image(uploaded_file)

    # Load trained model (make sure you saved it earlier with joblib.dump)
    with open("finalized_model.sav","rb") as f:
        model=pickle.load(f)
    # Predict
    prediction = model.predict(df_input)[0]

    st.success(f"Predicted Class: {prediction}")