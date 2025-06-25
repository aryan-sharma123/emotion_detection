


Gender-Specific Speech Emotion Recognition Web App

This project is a **speech emotion recognition (SER)** system that:
- Accepts `.wav` audio files as input
- Automatically detects the speaker's gender using a pretrained **ECAPA-TDNN model**
- Uses a gender-specific **CNN emotion classifier** to predict one of 8 emotions
- Is deployed as a **Streamlit web app**

---
Models Used

- **Gender Detection**:
  - Model: [`JaesungHuh/voice-gender-classifier`](https://huggingface.co/JaesungHuh/voice-gender-classifier)
  - Framework: `speechbrain`, `ECAPA-TDNN`
  
- **Emotion Classification**:
  - 2 separate **CNN models** trained using gender-specific feature distributions
  - Trained on features: MFCC, Chroma, Tonnetz, Spectral Contrast, Mel-Spectrogram, etc.
  - Output: 8 emotions

---

## 😃 Emotions Predicted

| Label | Emotion     |
|-------|-------------|
| 0     | Neutral     |
| 1     | Calm        |
| 2     | Happy       |
| 3     | Sad         |
| 4     | Angry       |
| 5     | Fearful     |
| 6     | Disgust     |
| 7     | Surprised   |

---

## 🛠️ Features Used

Extracted using `librosa`:
- MFCCs (mean, std, median)
- Chroma STFT
- Spectral Contrast
- Tonnetz
- Mel-Spectrogram (mean, std, max, median)
- Spectral Centroid, Bandwidth, Rolloff, Flatness
- Zero Crossing Rate
- RMS Energy

---




for male model predictions 
![image](https://github.com/user-attachments/assets/b5a55869-e5b5-40c4-a0f6-b7ab5ad4e7ee)

![image](https://github.com/user-attachments/assets/b7768ada-9335-4b36-a461-09627f735a7f)

for female mode predictions

![image](https://github.com/user-attachments/assets/2c785606-875b-4c5d-89b4-8a43f0680566)

![image](https://github.com/user-attachments/assets/2b583be5-0e8a-483d-88da-6ed14d847896)


