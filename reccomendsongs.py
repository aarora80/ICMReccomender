from tensorflow.keras.models import load_model
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt

def collect_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):  # Adjust file extension if needed
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
    return audio_files

archive_directory = 'archive'
audio_files = collect_audio_files(archive_directory)

# Load the trained model
model = load_model('emotion_classifier_model.h5')

def preprocess_audio_files(audio_files):
    features_list = []
    for file_path in audio_files:
        try:
            # Load audio file
            y, sr = librosa.load(file_path, duration=30)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            tempo = librosa.beat.tempo(y=y, sr=sr)
            
            # Aggregate features
            features = np.concatenate((np.mean(mfccs, axis=1), np.mean(chroma, axis=1), [tempo.mean()]))
            features_list.append(features)
            
        except Exception as e:
            print("Error encountered while parsing audio file:", file_path)
    
    return np.array(features_list)

def recommend_songs(audio_files, threshold=0.5):
    # Preprocess audio files
    X_preprocessed = preprocess_audio_files(audio_files)
    
    # Use the model to predict emotional categories
    predicted_probabilities = model.predict(X_preprocessed)
    
    # Define emotional categories
    emotional_categories = ['Devotional', 'Happy', 'Party', 'Romantic', 'Sad']
    
    # Recommend songs based on predicted probabilities
    recommended_songs = {}
    for i, category in enumerate(emotional_categories):
        if predicted_probabilities[:, i].max() > threshold:
            recommended_songs[category] = get_songs_from_category(category)
    
    return recommended_songs

def get_songs_from_category(category):
    songs = []
    category_directory = os.path.join(archive_directory, category, category)
    for root, dirs, files in os.walk(category_directory):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):  # Adjust file extensions if needed
                songs.append(os.path.join(root, file))
    return songs

def visualize_recommendations(recommendations):
    categories = list(recommendations.keys())
    counts = [len(recommendations[category]) for category in categories]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.title('Distribution of Songs Across Emotional Categories')
    plt.xlabel('Emotional Category')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45)
    plt.show()

recommendations = recommend_songs(audio_files)
print(recommendations)
# Call the function to visualize recommendations
visualize_recommendations(recommendations)