
# 🌍 Speech Accent Recognition using Deep Learning & Traditional ML

### Group 6 - Speech Recognition Systems
**University Project | North South University (NSU)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview
This project explores the classification of English speech accents using both **Traditional Machine Learning** and **State-of-the-Art Deep Learning** techniques. 

We classify speakers into **9 distinct geographical regions** (e.g., *Western European, South Asian, English, African*) using audio samples. Our study compares feature-based approaches (MFCCs + SVM/Random Forest) against end-to-end fine-tuning of the **Wav2Vec 2.0** Transformer model.

The final fine-tuned model achieves a **48.13% accuracy**, significantly outperforming traditional baselines (~29%) and the untrained base model (~4.5%).

---

## 📊 Dataset
We utilized the **Speech Accent Archive** dataset.
* **Source:** Crowd-sourced recordings of speakers reading the same paragraph ("Please call Stella...").
* **Total Samples:** ~2,138 audio files.
* **Classes:** Mapped into 9 broad regions:
    * *African, East Asian/SE Asian, Eastern European, English, Middle Eastern, Northern European, Oceanian/Other, South Asian, Western European.*

---

## 🧠 Methodology

### 1. Traditional Machine Learning (Baseline)
We extracted **MFCCs (Mel-Frequency Cepstral Coefficients)** from the audio to create a tabular dataset.
* **Feature Extraction:** Librosa (13 MFCCs, mean & variance).
* **Models:** k-NN, Random Forest, SVM, XGBoost.
* **Result:** Peaked at **29.44% Accuracy** (Random Forest).

### 2. Deep Learning (Fine-Tuning)
We fine-tuned a pre-trained **Wav2Vec 2.0 Base** model (`facebook/wav2vec2-base`) directly on the raw audio waveforms.
* **Architecture:** CNN Feature Encoder + Transformer Context Block + Classification Head.
* **Training:** * Optimizer: AdamW
    * Epochs: 15
    * Batch Size: 4 (Accumulated to 8)
    * **Augmentation:** Added Gaussian Noise to training data to improve robustness.
* **Hardware:** Trained on CPU (Intel/AMD) for stability.

---

## 🏆 Results & Performance

| Model | Accuracy | Macro F1 | Weighted F1 | Key Insight |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Untrained)** | 4.50% | 0.027 | 0.049 | Shows the model has zero zero-shot knowledge of these classes. |
| **k-NN** | 23.36% | 0.160 | 0.223 | Better than random, but struggles with complex audio features. |
| **SVM** | 23.60% | 0.196 | 0.240 | **Most Balanced:** Best performance on minority classes (African/Eastern European). |
| **XGBoost** | 26.64% | 0.164 | 0.252 | Strong performance on dominant classes but overfits. |
| **Random Forest** | 29.44% | 0.147 | 0.235 | **Best Traditional Model.** Good generalization on tabular MFCC data. |
| **Wav2Vec 2.0 (Fine-Tuned) 3 epoch** | **41.36%** | **0.221** | **0.430** | **Not bad.** Leaned but was having problem with white noises. |
| **Wav2Vec 2.0 (Fine-Tuned) 15 epoch** | **48.13%** | **0.259** | **0.445** | **Overall Winner.** The Transformer architecture successfully learned accent-specific phonemes and white noise removal. |

### Confusion Matrix (Wav2Vec 2.0)
The model excels at detecting **English** (F1: 0.85) and **Oceanian** (F1: 0.42) accents but demonstrates bias against underrepresented classes (e.g., Northern European), highlighting the need for balanced data or class weighting.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/speech-accent-recognition.git](https://github.com/your-username/speech-accent-recognition.git)
    cd speech-accent-recognition
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

### 1. Training the Model
To reproduce the 15-epoch fine-tuned model:
```bash
python train_accent_v2.py
````

*Note: This script forces CPU usage for stability. For GPU training, remove the `os.environ["CUDA_VISIBLE_DEVICES"]` line.*

### 2\. Evaluating the Model

To generate the Accuracy, F1 Scores, and Confusion Matrix:

```bash
python evaluate_finetuned.py
```

This will save `finetuned_results_table.csv` and `finetuned_confusion_matrix.png`.

### 3\. Inference (Predict on New Video/Audio)

To predict the accent of a specific video file (e.g., an interview):

```bash
python predict_video.py --path "path/to/video.mp4"
```

-----

## 🔮 Future Work

  * **Multilingual Models:** We plan to fine-tune **XLS-R (300M)**, which is pre-trained on 53 languages, to better capture non-English phonemes.
  * **Noise Robustness:** Experimenting with **WavLM** to handle background noise and overlapping speech better.
  * **Data Augmentation:** Implementing advanced augmentation (Background Noise, RIR Reverb) using `audiomentations`.

-----

## 👥 Contributors

  * **Ahmad Fahmid** - *overall project architecture, Random Forest implementation and fine-tuning*
  * **Fahim Foysal** - *XGBoost classifier development and fine-tuning*
  * **Shefa Tabassum** - *k-NN classifier with Gaussian noise augmentation and fine-tuning*
  * **Jannatul Ferdous Mim** - SVM classifier implementation
  * **Group of 4 Members - Group 6** - *North South University*

<!-- end list -->

```
```


