import torch
import librosa
import numpy as np
import os
import moviepy.editor as mp
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your video
VIDEO_PATH = r"C:\Users\alvi00\Desktop\Nsu\speech_recognization_systems-group-6\german.mp4"

# Path to your NEW Fine-Tuned Model
MODEL_PATH = "./wav2vec2_accent_finetuned_v2/checkpoint-3210" 
# (Note: check if 'checkpoint-642' exists, otherwise use './wav2vec2_accent_finetuned')

print(f"Loading Fine-Tuned Model from: {MODEL_PATH}...")

# ==========================================
# 2. LOAD MODEL
# ==========================================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

# Use CPU since we know GPU is tricky on your setup right now
device = "cpu" 
model = model.to(device)
print("✅ Model Loaded Successfully!")

# Get the class names from the model config
id2label = model.config.id2label

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================
def predict_accent(video_path):
    print(f"\n🎥 Processing Video: {video_path}")
    
    # A. Extract Audio
    audio_path = "temp_extracted_audio.wav"
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return

    # B. Load & Preprocess
    print("🔊 Analyzing Audio...")
    # Load 20 seconds of audio (max)
    speech, sr = librosa.load(audio_path, sr=16000, duration=20) 
    
    inputs = feature_extractor(
        speech, 
        sampling_rate=16000, 
        return_tensors="pt", 
        padding=True
    )

    # C. Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Calculate Probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get Top Prediction
    predicted_id = torch.argmax(probs, dim=-1).item()
    predicted_label = id2label[predicted_id]
    confidence = probs[0][predicted_id].item()

    # D. Output
    print("\n" + "="*40)
    print(f"🏆 PREDICTED ACCENT: {predicted_label.upper()}")
    print(f"🎯 Confidence: {confidence:.2%}")
    print("="*40)
    
    print("\nFull Probability Distribution:")
    # Sort and print all probabilities
    scores = [(id2label[i], probs[0][i].item()) for i in range(len(id2label))]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for label, score in scores:
        print(f"{label}: {score:.2%}")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

# ==========================================
# 4. RUN
# ==========================================
predict_accent(VIDEO_PATH)