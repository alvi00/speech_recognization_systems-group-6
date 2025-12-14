import pandas as pd
import numpy as np
import os
import torch
import librosa
import gc
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
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP - MEMORY OPTIMIZED
# ==========================================
BASE_PATH = r"C:\Users\Shefa\speech_recognization_systems-group-6"
CSV_PATH = os.path.join(BASE_PATH, "speakers_all.csv")
AUDIO_DIR = os.path.join(BASE_PATH, "recordings", "recordings") 

# Use a SMALL model for baseline to save memory
MODEL_ID = "facebook/wav2vec2-base"  # Smaller and faster than w2v-bert-2.0

# Force CPU and reduce memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)  # Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("="*60)
print("🚀 MEMORY-OPTIMIZED BASELINE EVALUATION")
print("="*60)
print(f"Using model: {MODEL_ID}")

# ==========================================
# 2. LOAD & MAP DATA - OPTIMIZED
# ==========================================
print(f"\n📂 Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Filter missing files
df['full_path'] = df['filename'].apply(lambda x: os.path.join(AUDIO_DIR, str(x) + ".mp3"))
df_exists = df['full_path'].apply(os.path.exists)
df = df[df_exists]

print(f"✅ Valid files found: {len(df)}")

if len(df) == 0:
    print("❌ NO FILES FOUND. Please check:")
    print(f"   Audio directory: {AUDIO_DIR}")
    print(f"   CSV has {len(df_exists)} rows, {sum(df_exists)} exist")
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

print(f"📊 Classes ({len(labels_list)}): {labels_list}")

# Show class distribution
print("\n📈 Class Distribution:")
class_counts = df['label'].value_counts()
for label, count in class_counts.items():
    print(f"   {label}: {count} samples")

# Take SMALL sample for memory constraints
SAMPLE_SIZE = 80  # Reduced for memory
df_sample = df.sample(SAMPLE_SIZE, random_state=42, weights=df['label'].map(lambda x: 1/class_counts[x]))  # Weighted sampling
print(f"\n📦 Using sample size: {len(df_sample)} files (weighted sampling)")

dataset = Dataset.from_pandas(df_sample[['full_path', 'label']])

def map_labels(batch):
    batch["labels"] = label2id[batch["label"]]
    return batch

dataset = dataset.map(map_labels, remove_columns=["label", "__index_level_0__"])

# ==========================================
# 3. PREPARE MODEL & FEATURES - OPTIMIZED
# ==========================================
print("\n⚙️ Loading feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

# Memory-optimized preprocessing
def preprocess_function(examples):
    paths = examples["full_path"]
    audio_arrays = []
    
    for path in paths:
        try:
            # Load only FIRST 2.5 SECONDS to save memory
            speech, sr = librosa.load(path, sr=16000, duration=2.5, mono=True)
            audio_arrays.append(speech)
        except Exception as e:
            # Add silent audio as fallback
            audio_arrays.append(np.zeros(int(16000 * 2.5)))
    
    # Use shorter max_length
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=int(16000 * 2.5),  # 2.5 seconds only
        truncation=True, 
        padding=True,
        return_tensors="np"
    )
    return inputs

print("🎵 Preprocessing audio (2.5 seconds each)...")
encoded_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    batch_size=4,  # Very small batch size
    remove_columns=["full_path"],
    desc="Processing audio"
)

# Clear memory
gc.collect()

print("🤖 Loading model...")
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(labels_list),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.eval()  # Set to evaluation mode

# ==========================================
# 4. ALTERNATIVE PREDICTION - MEMORY SAFE
# ==========================================
print("\n🎯 Running predictions (memory-safe method)...")

def predict_single_samples(model, dataset, batch_size=2):
    """Predict one small batch at a time to save memory"""
    all_predictions = []
    all_labels = []
    
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        # Get batch
        batch_indices = list(range(start_idx, end_idx))
        batch = dataset.select(batch_indices)
        
        # Convert to tensors
        inputs = torch.tensor(batch["input_values"], dtype=torch.float32)
        labels = torch.tensor(batch["labels"], dtype=torch.long)
        
        # Predict with no gradient
        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())
        
        # Clear memory
        del inputs, outputs, predictions
        gc.collect()
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            print(f"   Processed {end_idx}/{num_samples} samples")
    
    return np.array(all_predictions), np.array(all_labels)

# Run prediction
y_pred, y_test = predict_single_samples(model, encoded_dataset, batch_size=2)

# ==========================================
# 5. GENERATE REPORT
# ==========================================
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*60)
print("📊 BASELINE MODEL PERFORMANCE (Untrained)")
print("="*60)
print(f"✅ Accuracy:      {acc:.2%}")
print(f"⚖️ Macro F1:      {f1_macro:.4f}")
print(f"🌍 Weighted F1:   {f1_weighted:.4f}")
print("="*60)

# Save Detailed Table
target_names = [id2label[i] for i in sorted(list(set(y_test) | set(y_pred)))]
report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

csv_filename = "baseline_results_optimized_shefa.csv"
df_report.to_csv(csv_filename)
print(f"\n💾 Results saved to: {csv_filename}")

# Show per-class performance
print("\n📋 Per-Class Performance:")
print("-" * 80)
for label_id, label_name in id2label.items():
    if label_id in y_test:
        mask = y_test == label_id
        if np.sum(mask) > 0:
            accuracy = np.mean(y_pred[mask] == y_test[mask])
            count = np.sum(mask)
            print(f"   {label_name:<45} Acc: {accuracy:.2%} ({count} samples)")

# ==========================================
# 6. PLOT CONFUSION MATRIX
# ==========================================
print("\n🎨 Generating visualization...")
plt.figure(figsize=(12, 10))
unique_labels = sorted(list(set(y_test) | set(y_pred)))
labels_text = [id2label[i] for i in unique_labels]

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

# Normalize for better visualization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_text,
            yticklabels=labels_text,
            ax=axes[0])
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].tick_params(axis='x', rotation=45)

# Plot normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels_text,
            yticklabels=labels_text,
            ax=axes[1])
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_title('Confusion Matrix (Normalized)')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Baseline Model Performance - Untrained wav2vec2-base', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("baseline_confusion_matrix_optimized.png", dpi=150, bbox_inches='tight')
print("📊 Confusion matrix saved to: baseline_confusion_matrix_optimized.png")
plt.show()

# ==========================================
# 7. ADDITIONAL ANALYSIS
# ==========================================
print("\n" + "="*60)
print("📈 ADDITIONAL ANALYSIS")
print("="*60)

# Most confused pairs
print("\n🔍 Most Confused Class Pairs:")
confusion_pairs = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((cm[i, j], labels_text[i], labels_text[j]))

confusion_pairs.sort(reverse=True, key=lambda x: x[0])

for count, true_label, pred_label in confusion_pairs[:10]:  # Top 10
    print(f"   {true_label} → {pred_label}: {count} samples")

# Random chance baseline
print(f"\n🎲 Random Chance Baseline: {1/len(labels_list):.2%}")
print(f"📈 Model vs Random: {acc - 1/len(labels_list):+.2%}")

print("\n" + "="*60)
print("✅ BASELINE EVALUATION COMPLETE")
print("="*60)