import pandas as pd
import numpy as np
import os
import torch
import librosa
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_PATH = r"E:\445 project\speech_recognization_systems-group-6\Speech_Dataset"
CSV_PATH = os.path.join(BASE_PATH, "speakers_all.csv")
AUDIO_DIR = os.path.join(BASE_PATH, "recordings", "recordings") 

MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
OUTPUT_DIR = "./wav2vec2_xlsr53_accent_finetuned"  # Saved to this folder

# 🛑 FORCE CPU (Safety for your setup)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(f"⚠️ FORCE CPU MODE: Training will take longer but is stable.")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df['full_path'] = df['filename'].apply(lambda x: os.path.join(AUDIO_DIR, str(x) + ".mp3"))
df = df[df['full_path'].apply(os.path.exists)]
print(f"✅ Found {len(df)} valid audio files.")

def get_region(native_language):
    lang = str(native_language).lower()
    if lang in ['english']: return "English"
    if lang in ['mandarin', 'cantonese', 'japanese', 'korean', 'thai', 'vietnamese', 'burmese']: return "East Asian or South East Asian"
    if lang in ['hindi', 'bengali', 'tamil', 'urdu', 'punjabi', 'gujarati', 'nepali', 'sinhalese']: return "South Asian"
    if lang in ['arabic', 'farsi', 'turkish', 'hebrew', 'pashto', 'kurdish', 'armenian']: return "Middle Eastern, Central Asian or Southern Europe"
    if lang in ['spanish', 'french', 'german', 'italian', 'portuguese', 'dutch']: return "Western European"
    if lang in ['russian', 'polish', 'ukrainian', 'croatian', 'czech', 'serbian', 'bulgarian']: return "Eastern European"
    if lang in ['swedish', 'norwegian', 'danish', 'finnish', 'icelandic']: return "Northern European"
    if lang in ['amharic', 'swahili', 'yoruba', 'igbo', 'hausa', 'twi', 'zulu']: return "African"
    return "Oceanian or Other" 

df['label'] = df['native_language'].apply(get_region)
labels_list = sorted(df['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}

dataset = Dataset.from_pandas(df[['full_path', 'label']])
def map_labels(batch):
    batch["labels"] = label2id[batch["label"]]
    return batch
dataset = dataset.map(map_labels, remove_columns=["label"])

# SPLIT FIRST, PROCESS LATER (noise only on train)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# ==========================================
# 3. FEATURE EXTRACTION & AUGMENTATION
# ==========================================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

# --- A. NOISY PREPROCESSING (For Training) ---
def preprocess_train(examples):
    paths = examples["full_path"]
    audio_arrays = []
    for path in paths:
        speech, sr = librosa.load(path, sr=16000)
        
        # 🔊 50% chance to add white noise
        if np.random.rand() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.amax(speech)
            noise = noise_amp * np.random.normal(size=speech.shape[0])
            speech = speech + noise
            
        audio_arrays.append(speech)
        
    return feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 4,  # 4 seconds max
        truncation=True, 
        padding=True
    )

# --- B. CLEAN PREPROCESSING (For Testing) ---
def preprocess_test(examples):
    paths = examples["full_path"]
    audio_arrays = []
    for path in paths:
        speech, sr = librosa.load(path, sr=16000)
        audio_arrays.append(speech)
        
    return feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 4, 
        truncation=True, 
        padding=True
    )

print("🔊 Processing Training Set (Adding Noise)...")
dataset["train"] = dataset["train"].map(preprocess_train, batched=True, batch_size=10, remove_columns=["full_path"])

print("✨ Processing Test Set (Keeping Clean)...")
dataset["test"] = dataset["test"].map(preprocess_test, batched=True, batch_size=10, remove_columns=["full_path"])

# ==========================================
# 4. TRAINING SETUP
# ==========================================
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(labels_list),
    label2id=label2id,
    id2label=id2label
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=2, 
    per_device_eval_batch_size=4,
    num_train_epochs=8,            
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    use_cpu=True,
    save_total_limit=2,
    dataloader_num_workers=0,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

# ==========================================
# 5. START TRAINING (WITH RESUME SUPPORT)
# ==========================================
print("\n🚀 Starting IMPROVED Training on CPU...")
print("⏳ This will take significantly longer (8 Epochs).")
print("   The model is now learning to ignore noise!")

# This line is the key: automatically resumes from the latest checkpoint if one exists
trainer.train(resume_from_checkpoint=True)

print(f"\n🎉 Training complete! Best model saved to {OUTPUT_DIR}")