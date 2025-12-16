import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Primary logic
def main():
    st.title("🧠 Brain Tumor Detection System")
    st.subheader("AI Powered Tumor Detection using Deep Learning")
    
    st.write("Upload a brain MRI image to detect if a tumor is present.")
    
    file = st.file_uploader("Please upload an MRI Image", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        
        # Preprocessing mainly for display and consistency
        image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
        
        if st.button("Detect Tumor"):
            st.write("Analysing...")
            
            try:
                model = load_model()
                
                # Preprocess for model
                img_array = np.array(image.convert('RGB'))
                img_array = img_array / 255.0  # Normalize
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
                
                prediction = model.predict(img_array)
                
                # Assuming class 0 is 'No Tumor' and class 1 is 'Tumor' (Alphabetical order usually: no, yes)
                # But need to match training generator indices. ImageDataGenerator sorts alphanumerically.
                # no -> 0, yes -> 1
                
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)
                
                if class_index == 1:
                    st.error(f"Prediction: TUMOR DETECTED ⚠️")
                    st.write(f"Confidence: {confidence:.2f}%")
                else:
                    st.success(f"Prediction: NO TUMOR DETECTED ✅")
                    st.write(f"Confidence: {confidence:.2f}%")
                    
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")
                st.info("Make sure 'model.h5' exists. Run 'python train_model.py' first.")

if __name__ == "__main__":
    main()
