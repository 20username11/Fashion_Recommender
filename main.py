import streamlit as st
import os
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle

# Load precomputed features and filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model, GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Page title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Find Your Similar Style</h1>", unsafe_allow_html=True)
st.sidebar.title("Navigation")
st.sidebar.write("Upload an image to find similar styles!")

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.image(display_image, caption="Uploaded Image")

        features = extract_features(os.path.join('uploads', uploaded_file.name), model)
        indices = recommend(features, feature_list)

        # Display similar images in a grid with captions
        st.subheader("Top 5 Similar Styles")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.image(filenames[indices[0][i]], caption=f"Style {i+1}", use_container_width=True)
    else:
        st.error("Error in saving the uploaded file. Please try again.")
