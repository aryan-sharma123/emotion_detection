import os
import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import numpy as np
import librosa
import joblib
from keras.models import load_model
import torchaudio
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

int_to_emotion = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}



## for locally do these 
# model_female = load_model(r"female_models\female_model.keras")
# model_male = load_model("male_models\\male_model.keras")
# scaler_female = joblib.load("female_models\\female_scaler.pkl")
# scaler_male = joblib.load("male_models\\male_scaler.pkl")

## for streamlit do these 


from tensorflow.keras.models import load_model


model_female = load_model("emotion_classifier_final/female_models/female_model.keras")
model_male = model("emotion_classifier_final/male_models/male_model.keras")
scaler_female = joblib.load("emotion_classifier_final/female_models/female_scaler.pkl")
scaler_male = joblib.load("emotion_classifier_final/male_models/male_scaler.pkl")


from voice_gender_classifier.model import ECAPA_gender

gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
gender_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_model.to(device)

def predict_gender(file):
    with torch.no_grad():
        return gender_model.predict(file, device=device)



# --- Your Feature Extraction Function ---
def extract_features(audio, sr, n_mfcc=40, n_chroma=12, n_bands=6):
    stft = np.abs(librosa.stft(audio))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.vstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1), np.median(mfccs, axis=1)])
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=n_chroma)
    chroma_processed = np.vstack([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=n_bands)
    contrast_processed = np.vstack([np.mean(contrast, axis=1), np.std(contrast, axis=1)])
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    tonnetz_processed = np.vstack([np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)])
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio)
    spectral_features = np.array([np.mean(centroid), np.std(centroid), np.mean(bandwidth), np.std(bandwidth),
                                  np.mean(rolloff), np.std(rolloff), np.mean(flatness), np.std(flatness)])
    zero_crossing = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    temporal_features = np.array([np.mean(zero_crossing), np.std(zero_crossing), np.mean(rms), np.std(rms)])
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_processed = np.array([np.mean(mel), np.std(mel), np.median(mel), np.max(mel)])
    return np.hstack([mfccs_processed.flatten(), chroma_processed.flatten(), contrast_processed.flatten(),
                      tonnetz_processed.flatten(), spectral_features, temporal_features, mel_processed])



# --- Streamlit UI ---
st.title("🎙️ Emotion Detection App")
st.markdown("Upload a `.wav` file. The system will detect gender and predict the emotion.")

uploaded_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save to temp file for both torchaudio and librosa use
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Gender prediction
    gender = predict_gender(temp_path)
    st.write(f"**Predicted Gender:** `{gender}`")
    
    # Feature extraction
    
    
    # Choose model
    if gender == 'female':
        audio, sr = librosa.load(temp_path, sr=16000)
        features = extract_features(audio, sr)
        features_scaled = scaler_female.transform([features])
        features_input = features_scaled.reshape((1, features_scaled.shape[1], 1))  # CNN shape
        probs = model_female.predict(features_input)[0]
        predicted_class = int_to_emotion[np.argmax(probs)]
    else:
        audio, sr = librosa.load(temp_path, sr=16000)
        features = extract_features(audio, sr)
        features_scaled = scaler_male.transform([features])
        features_input = features_scaled.reshape((1, features_scaled.shape[1], 1))  # CNN shape
        probs = model_male.predict(features_input)[0]
        predicted_class = int_to_emotion[np.argmax(probs)]
    # predicted_class = label_encoder.inverse_transform([np.argmax(probs)])[0]
    
    st.success(f"**Predicted Emotion:** `{predicted_class}`")
    st.bar_chart(probs)

    # Cleanup
    os.remove(temp_path)
