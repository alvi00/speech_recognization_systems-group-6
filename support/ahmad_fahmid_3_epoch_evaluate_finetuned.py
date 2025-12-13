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
# 1. SETUP & PATHS
# ==========================================
BASE_PATH = r"C:\Users\alvi00\Desktop\Nsu\speech_recognization_systems-group-6\Speech_Dataset"
CSV_PATH = os.path.join(BASE_PATH, "speakers_all.csv")
AUDIO_DIR = os.path.join(BASE_PATH, "recordings", "recordings") 

# POINT TO YOUR SAVED MODEL
MODEL_PATH = "./wav2vec2_accent_finetuned/checkpoint-642"

# Force CPU (Safe mode)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print(f"📂 Loading Fine-Tuned Model from: {MODEL_PATH}")

# ==========================================
# 2. LOAD & PREPARE DATA (Same as Training)
# ==========================================
print("⏳ Loading dataset to recreate Test Split...")
df = pd.read_csv(CSV_PATH)

# Filter files
df['full_path'] = df['filename'].apply(lambda x: os.path.join(AUDIO_DIR, str(x) + ".mp3"))
df = df[df['full_path'].apply(os.path.exists)]

# Map Regions
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

dataset = dataset.map(map_labels, remove_columns=["label", "__index_level_0__"])

# CRITICAL: Use the SAME seed (42) to get the SAME test set as training
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

print(f"✅ Test Set Created: {len(test_dataset)} samples")

# ==========================================
# 3. PREPROCESS (Librosa)
# ==========================================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

def preprocess_function(examples):
    paths = examples["full_path"]
    audio_arrays = []
    for path in paths:
        speech, sr = librosa.load(path, sr=16000)
        audio_arrays.append(speech)
        
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 4, 
        truncation=True, 
        padding=True
    )
    return inputs

print("⏳ Preprocessing Test Data (this takes a moment)...")
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=10, remove_columns=["full_path"])

# ==========================================
# 4. LOAD MODEL & PREDICT
# ==========================================
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./tmp_eval_finetuned", 
        per_device_eval_batch_size=8,
        use_cpu=True,
        report_to="none"
    ),
    processing_class=feature_extractor,
)

print("\n🚀 Running Final Evaluation on Test Set...")
predictions = trainer.predict(encoded_test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_test = predictions.label_ids

# ==========================================
# 5. GENERATE REPORT
# ==========================================
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*40)
print("🏆 FINE-TUNED MODEL PERFORMANCE")
print("="*40)
print(f"✅ Accuracy:      {acc:.2%}")
print(f"⚖️ Macro F1:      {f1_macro:.4f}")
print(f"🌍 Weighted F1:   {f1_weighted:.4f}")
print("="*40)

# Save Detailed Table
target_names = [id2label[i] for i in sorted(list(set(y_test) | set(y_pred)))]
report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

csv_filename = "finetuned_results_table.csv"
df_report.to_csv(csv_filename)
print(f"\n💾 Results table saved to: {csv_filename}")

# ==========================================
# 6. PLOT CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(12, 10))
unique_labels = sorted(list(set(y_test) | set(y_pred)))
labels_text = [id2label[i] for i in unique_labels]

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=labels_text,
            yticklabels=labels_text)
plt.xlabel('Predicted Accent')
plt.ylabel('Actual Accent')
plt.title('Confusion Matrix - Fine-Tuned Wav2Vec2')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("finetuned_confusion_matrix.png") # Save it automatically
plt.show()