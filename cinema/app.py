import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import uuid
import os

# Set page configuration
st.set_page_config(page_title="Movie Hit Predictor", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #ddd;
    }
    .title { color: #2c3e50; font-size: 36px; font-weight: bold; text-align: center; }
    .subtitle { color: #34495e; font-size: 20px; text-align: center; margin-bottom: 30px; }
    .result { font-size: 24px; font-weight: bold; text-align: center; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    if not os.path.exists('movies_extended.csv'):
        st.error("Dataset 'movies_extended.csv' not found. Please upload the file.")
        st.stop()
    df = pd.read_csv('movies_extended.csv')
    df[['rating', 'year', 'runtime']] = df[['rating', 'year', 'runtime']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df_genres = df.assign(genres=df['genres'].str.split('|')).explode('genres').dropna(subset=['genres'])
    all_genres = sorted(df_genres['genres'].unique())
    return df, all_genres

df, all_genres = load_data()

# Load or rebuild model
@st.cache_resource
def load_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    try:
        model.load_weights('model.weights.h5')  # Load weights with correct extension
    except:
        st.warning("Model weights not found. Please ensure 'model.weights.h5' is in the directory.")
    return model

# Prepare model
input_shape = len(all_genres) + 3  # Genres + rating, runtime, year
model = load_model(input_shape)

# UI Layout
st.markdown('<div class="title">Movie Hit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict if a movie will be a Hit or Flop based on its genres, rating, runtime, and year!</div>', unsafe_allow_html=True)

# Input form
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        genres_input = st.multiselect(
            "Select Genres",
            options=all_genres,
            help="Choose one or more genres for the movie."
        )
        rating_input = st.slider(
            "Rating (1.0 - 10.0)",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Select the movie's rating."
        )
    
    with col2:
        runtime_input = st.slider(
            "Runtime (minutes)",
            min_value=60,
            max_value=240,
            value=120,
            step=1,
            help="Select the movie's runtime in minutes."
        )
        year_input = st.number_input(
            "Year",
            min_value=1900,
            max_value=2030,
            value=2023,
            step=1,
            help="Enter the movie's release year."
        )
    
    submit_button = st.form_submit_button(label="Predict")

# Prediction logic
if submit_button:
    # Prepare input data
    genre_features = np.zeros(len(all_genres))
    for g in genres_input:
        if g in all_genres:
            genre_features[all_genres.index(g)] = 1
    
    input_data = np.append(genre_features, [
        (rating_input - df['rating'].mean()) / df['rating'].std(),
        (runtime_input - df['runtime'].mean()) / df['runtime'].std(),
        (year_input - df['year'].mean()) / df['year'].std()
    ]).astype(np.float32)
    
    # Make prediction
    pred = model.predict(np.array([input_data]), verbose=0)[0][0]
    result = "Hit!" if pred > 0.5 else "Flop."
    
    # Display result
    st.markdown(f'<div class="result">Prediction: {result}</div>', unsafe_allow_html=True)
    st.write(f"Details: Genres: {', '.join(genres_input)}, Rating: {rating_input}, Runtime: {runtime_input} min, Year: {year_input}")
    st.write(f"Probability of being a Hit: {pred:.2%}")

# Save model (for reference, assuming the original script is modified to save)
# This is not executed in Streamlit but included for completeness
"""
model.save_weights('model.weights.h5')
"""