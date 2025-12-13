import pandas as pd
import numpy as np
import os
import torch
import librosa
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    Trainer, 
    TrainingArguments
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP
# ==========================================
BASE_PATH = r"C:\Users\alvi00\Desktop\Nsu\speech_recognization_systems-group-6\Speech_Dataset"
CSV_PATH = os.path.join(BASE_PATH, "speakers_all.csv")
AUDIO_DIR = os.path.join(BASE_PATH, "recordings", "recordings") 

MODEL_ID = "facebook/wav2vec2-base"

# Force CPU for safety (since this is just a quick check and your GPU is new)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ==========================================
# 2. LOAD & MAP DATA
# ==========================================
print(f"Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Filter missing files
df['full_path'] = df['filename'].apply(lambda x: os.path.join(AUDIO_DIR, str(x) + ".mp3"))
df = df[df['full_path'].apply(os.path.exists)]

print(f"✅ Valid files found: {len(df)}")

if len(df) == 0:
    print("❌ STILL NO FILES FOUND. Please check the path manually.")
    exit()

# --- REGION MAPPING ---
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

print(f"Classes: {labels_list}")

# Sample 200 files for quick baseline check
df_sample = df.sample(200, random_state=42) 

dataset = Dataset.from_pandas(df_sample[['full_path', 'label']])

def map_labels(batch):
    batch["labels"] = label2id[batch["label"]]
    return batch

dataset = dataset.map(map_labels, remove_columns=["label", "__index_level_0__"])

# ==========================================
# 3. PREPARE MODEL & FEATURES
# ==========================================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

# Robust Librosa Preprocessing (fixes the torchcodec crash)
def preprocess_function(examples):
    paths = examples["full_path"]
    audio_arrays = []
    for path in paths:
        speech, sr = librosa.load(path, sr=16000)
        audio_arrays.append(speech)
        
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 5, 
        truncation=True, 
        padding=True
    )
    return inputs

print("Preprocessing audio (Using Librosa)...")
encoded_dataset = dataset.map(preprocess_function, batched=True, batch_size=10, remove_columns=["full_path"])

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(labels_list),
    label2id=label2id,
    id2label=id2label
)

# ==========================================
# 4. PREDICT & GENERATE REPORT
# ==========================================
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./tmp_eval", 
        per_device_eval_batch_size=8,
        use_cpu=True, # Safety first for baseline check
        report_to="none"
    ),
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

print("\n⏳ Running Baseline Predictions...")
# Get raw predictions
predictions = trainer.predict(encoded_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_test = predictions.label_ids

# --- METRICS CALCULATION ---
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*40)
print("📊 BASELINE MODEL PERFORMANCE (Untrained)")
print("="*40)
print(f"❌ Accuracy:      {acc:.2%}")
print(f"⚖️ Macro F1:      {f1_macro:.4f}")
print(f"🌍 Weighted F1:   {f1_weighted:.4f}")
print("="*40)

# --- SAVE RESULTS TABLE ---
target_names = [id2label[i] for i in sorted(list(set(y_test) | set(y_pred)))]
report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

csv_filename = "baseline_results_table.csv"
df_report.to_csv(csv_filename)
print(f"\n💾 Baseline results saved to: {csv_filename}")

# --- PLOT CONFUSION MATRIX ---
plt.figure(figsize=(12, 10))
unique_labels = sorted(list(set(y_test) | set(y_pred)))
labels_text = [id2label[i] for i in unique_labels]

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=labels_text,
            yticklabels=labels_text)
plt.xlabel('Predicted Accent (Random)')
plt.ylabel('Actual Accent')
plt.title('Confusion Matrix - Baseline (Untrained)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()