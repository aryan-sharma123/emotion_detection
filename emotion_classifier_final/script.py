import os
import numpy as np
import librosa
import joblib
import pandas as pd
import torch
from keras.models import load_model
from voice_gender_classifier.model import ECAPA_gender
from tqdm import tqdm

# Emotion mapping
int_to_emotion = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}

# Load models and scalers
model_female = load_model("female_models/female_model.keras")
model_male = load_model("male_models/male_model.keras")
scaler_female = joblib.load("female_models/female_scaler.pkl")
scaler_male = joblib.load("male_models/male_scaler.pkl")

# Load gender classification model
gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
gender_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_model.to(device)

def predict_gender(file_path):
    with torch.no_grad():
        return gender_model.predict(file_path, device=device)

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

# ----------- MAIN --------------
def process_folder(folder_path):
    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path.")
        return

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not wav_files:
        print("⚠️ No `.wav` files found in the folder.")
        return

    results = []

    print(f"\n🔍 Found {len(wav_files)} files. Processing...\n")

    for filename in tqdm(wav_files):
        file_path = os.path.join(folder_path, filename)

        try:
            gender = predict_gender(file_path)
            audio, sr = librosa.load(file_path, sr=16000)
            features = extract_features(audio, sr)

            if gender == 'female':
                features_scaled = scaler_female.transform([features])
                features_input = features_scaled.reshape((1, features_scaled.shape[1], 1))
                probs = model_female.predict(features_input)[0]
            else:
                features_scaled = scaler_male.transform([features])
                features_input = features_scaled.reshape((1, features_scaled.shape[1], 1))
                probs = model_male.predict(features_input)[0]

            predicted_emotion = int_to_emotion[np.argmax(probs)]
            results.append((filename, gender, predicted_emotion))

        except Exception as e:
            results.append((filename, "Error", str(e)))

    df = pd.DataFrame(results, columns=["Filename", "Gender", "Predicted Emotion"])
    output_csv = os.path.join(folder_path, "emotion_predictions.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}\n")
    print(df)


if __name__ == "__main__":
    folder = input("Enter the path to folder containing .wav files: ").strip()
    process_folder(folder)
    print(result)
