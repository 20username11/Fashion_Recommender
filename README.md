# Fashion Recommender System

This project is a **Fashion Recommender System** that recommends visually similar fashion styles based on a user-uploaded image. It leverages a pre-trained **ResNet50** model to extract image features and find the most relevant matches from a dataset.

## Table of Contents
- [Features](#features)
- [Core Components](#core-components)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Future Enhancements](#future-enhancements)

## Features
- **Pre-trained Model**: Uses ResNet50 for feature extraction.
- **Efficient Search**: Finds the top 5 most similar images using the Nearest Neighbors algorithm.
- **Interactive UI**: Provides an easy-to-use Streamlit interface for uploading images and viewing recommendations.

## Core Components

### app.py
- Extracts features from dataset images.
- Saves the precomputed embeddings (`embeddings.pkl`) and filenames (`filenames.pkl`).

### main.py
- Implements the Streamlit interface.
- Loads precomputed embeddings and filenames for efficient recommendations.
- Provides a recommendation system based on user-uploaded images.

## How It Works

1. **Feature Extraction**:
   - Images are resized to `224x224` and preprocessed using ResNet50.
   - Extracted features are normalized using L2 norm.

2. **Recommendation**:
   - The Nearest Neighbors algorithm (Euclidean distance) identifies the closest matches.

3. **Display**:
   - Displays the uploaded image and top 5 similar images in a grid layout.

## Setup and Installation

### Prerequisites
- Python 3.8.10
- A TensorFlow-compatible environment

### Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   ```
   Activate the environment:
   - **Windows**:
     ```bash
     .\env\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source env/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Precompute Embeddings (Optional)**:
   - If embeddings need to be regenerated:
     ```bash
     python app.py
     ```

5. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Upload a fashion image using the file uploader in the Streamlit app.
2. View the uploaded image alongside the top 5 most similar styles from the dataset.
## Dependencies

The project dependencies are listed in `requirements.txt`:

```txt
numpy
tensorflow
streamlit
Pillow
scikit-learn
tqdm
```
Install them with:
```bash
pip install -r requirements.txt
```

