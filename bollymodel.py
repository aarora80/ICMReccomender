import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping




# Function to extract features from audio files
def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tempo = librosa.beat.tempo(y=y, sr=sr)
        
        # Aggregate features
        features = np.concatenate((np.mean(mfccs, axis=1), np.mean(chroma, axis=1), [tempo.mean()]))
        
    except Exception as e:
        print("Error encountered while parsing audio file:", file_path)
        return None
    
    return features

# Path to the directory where the audio files are stored
dataset_path = 'archive'  # Adjust this path if needed

# List of emotions or categories in your dataset
emotions = ['Devotional', 'Happy', 'Party', 'Romantic', 'Sad']

# Extract features and labels
feature_list = []
# Loop through each emotion folder
for emotion in emotions:
    emotion_path = os.path.join(dataset_path, emotion, emotion)
    for root, dirs, files in os.walk(emotion_path):
        for file in files:
            file_path = os.path.join(root, file)
            features = extract_features(file_path)
            if features is not None:
                feature_list.append({'Emotion': emotion, 'File': file, 'Features': features})

df = pd.DataFrame(feature_list)

print(df.head())

# Prepare Data
X = np.array(df['Features'].tolist())
y = df['Emotion']

# Encode categorical labels into numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Data prepared and split successfully!")

# Reshape features for input to CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Build CRNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotions), activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save model
model.save('emotion_classifier_model.h5')