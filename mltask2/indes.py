import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
# Define em
EMOTION_LABELS = {
    "03-01-01-01-01-01-24.wav": "anger",
    "03-01-02-01-01-01-24.wav": "happy",
    "03-01-03-01-01-01-24.wav": "sad",
    "03-01-04-01-01-01-24.wav": "neutral",
    "03-01-05-01-01-01-24.wav": "fear",
    "03-01-06-01-01-01-24.wav": "disgust",
    "03-01-07-01-01-01-24.wav": "surprise",
    "03-01-08-01-01-01-24.wav": "boredom",
    "03-01-01-01-02-01-24.wav": "anger",
    "03-01-02-01-02-01-24.wav": "happy",
    "03-01-03-01-02-01-24.wav": "sad",
    "03-01-04-01-02-01-24.wav": "neutral",
    "03-01-05-01-02-01-24.wav": "fear",
    "03-01-06-01-02-01-24.wav": "disgust",
    "03-01-07-01-02-01-24.wav": "surprise",
    "03-01-08-01-02-01-24.wav": "boredom",
    # Add additional mappings based on your dataset
}



# Feature extraction function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        hop_length = 512

        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # MFCC and delta features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)

        # Beat-synchronous MFCC and delta features
        _, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

        # Chroma features
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

        # Combine features
        features = np.hstack([beat_chroma.flatten(), beat_mfcc_delta.flatten()])
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Prepare the dataset
def prepare_dataset(audio_dir):
    features, labels = [], []
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(audio_dir, file_name)
            emotion = EMOTION_LABELS.get(file_name)
            if emotion:
                print(f"Processing file: {file_name} with label: {emotion}")
                feature = extract_features(file_path)
                if feature is not None:
                    # Ensure consistent feature size
                    max_length = 500  # Adjust max length as needed
                    padded_feature = np.pad(feature, (0, max(0, max_length - len(feature))), mode='constant')
                    features.append(padded_feature[:max_length])
                    labels.append(emotion)
                else:
                    print(f"Skipping file due to feature extraction error: {file_name}")
            else:
                print(f"File {file_name} does not have a label in EMOTION_LABELS.")
    return np.array(features), np.array(labels)

# Directory path where the audio files are stored
audio_dir = 'C:/local_d_data/6thsmester/machinelearning/Actor_24'

# Prepare dataset
X, y = prepare_dataset(audio_dir)

# Check if dataset is empty
if X.size == 0 or y.size == 0:
    print("Error: No valid data found. Check your directory and EMOTION_LABELS.")
    exit()
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize feature importance
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[sorted_indices])
plt.title("Feature Importance (Sorted)")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.show()
