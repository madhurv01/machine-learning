import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess
df = pd.read_csv('movies_extended.csv')
df[['rating', 'year', 'runtime']] = df[['rating', 'year', 'runtime']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
ratings = df['rating']
df_genres = df.assign(genres=df['genres'].str.split('|')).explode('genres').dropna(subset=['genres'])
all_genres = sorted(df_genres['genres'].unique())

# Preprocess features
features = pd.get_dummies(df_genres['genres'], prefix='genre').groupby(df_genres.index).sum().join(
    df[['rating', 'runtime', 'year']].apply(lambda x: (x - x.mean()) / x.std())
).reindex(columns=[f'genre_{g}' for g in all_genres] + ['rating', 'runtime', 'year'], fill_value=0)
X = features.values.astype(np.float32)
y = (ratings > 7.5).astype(np.int32).values

# Build and train model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, verbose=0)
model.save_weights('model.weights.h5')  # Save the model weights with correct extension

# Sample prediction
genre_features = np.zeros(len(all_genres))
for g in 'Comedy,Drama'.split(','):
    if g in all_genres:
        genre_features[all_genres.index(g)] = 1
input_data = np.append(genre_features, [
    (7.0 - df['rating'].mean()) / df['rating'].std(),
    (120 - df['runtime'].mean()) / df['runtime'].std(),
    (2023 - df['year'].mean()) / df['year'].std()
]).astype(np.float32)
pred = model.predict(np.array([input_data]), verbose=0)[0][0]
print(f"Sample prediction for Comedy,Drama, rating 7.0, 120 min, 2023: {'Hit!' if pred > 0.5 else 'Flop.'}")

# User input prediction
genres_input = input("Enter genres (comma-separated, e.g., Comedy,Drama): ").strip()
rating_input = float(input("Enter rating (1.0-10.0, e.g., 7.0): "))
runtime_input = float(input("Enter runtime (minutes, e.g., 120): "))
year_input = float(input("Enter year (e.g., 2023): "))
genre_features = np.zeros(len(all_genres))
for g in genres_input.split(','):
    g = g.strip()
    if g in all_genres:
        genre_features[all_genres.index(g)] = 1
input_data = np.append(genre_features, [
    (rating_input - df['rating'].mean()) / df['rating'].std(),
    (runtime_input - df['runtime'].mean()) / df['runtime'].std(),
    (year_input - df['year'].mean()) / df['year'].std()
]).astype(np.float32)
pred = model.predict(np.array([input_data]), verbose=0)[0][0]
print(f"Prediction for {genres_input}, rating {rating_input}, {runtime_input} min, {year_input}: {'Hit!' if pred > 0.5 else 'Flop.'}")