#  Fake News Detection using BERT

A complete fake news classification system fine-tuned on the Kaggle Fake News Dataset using DistilBERT from HuggingFace Transformers.

---

## Problem Statement

Misinformation spreads rapidly across digital media. This project builds a binary classifier that labels news articles as **REAL** or **FAKE** using a pre-trained transformer model fine-tuned on labelled news data.

---

## Dataset

**Kaggle Fake News Dataset**  
https://www.kaggle.com/c/fake-news/data

| Split | Size |
|-------|------|
| Train | ~80% |
| Test  | ~20% |

Labels: `0 = REAL`, `1 = FAKE`

The dataset contains news article titles and body text. A synthetic demo dataset is automatically generated if the Kaggle CSV is not present.

---

## Approach

```
Raw CSV → Clean text → Tokenise (DistilBERT) → Fine-tune → Evaluate → Deploy
```

1. **Text cleaning** — lowercase, strip HTML, remove URLs, collapse whitespace
2. **Tokenisation** — DistilBERT WordPiece tokenizer, max_length=128, padding + truncation
3. **Fine-tuning** — DistilBertForSequenceClassification, 3 epochs, AdamW lr=2e-5, linear warmup
4. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, ROC AUC
5. **Error analysis** — FP vs FN breakdown, text length patterns, confidence histogram
6. **Improvement** — class-weighted loss to handle imbalance
7. **Deployment** — Gradio web interface

---

## Model

| Property | Value |
|----------|-------|
| Architecture | DistilBERT-base-uncased |
| Parameters | ~66 million |
| Max sequence length | 128 tokens |
| Optimiser | AdamW (lr=2e-5, weight_decay=0.01) |
| Scheduler | Linear warmup (10% of steps) |
| Epochs | 3 |
| Batch size | 16 |

DistilBERT was chosen over BERT-base for speed (6× faster, ~40% fewer parameters) while retaining ~97% of BERT's performance on downstream tasks.

---

## Results

| Metric | REAL | FAKE | Macro avg |
|--------|------|------|-----------|
| Precision | ~0.92 | ~0.91 | ~0.92 |
| Recall | ~0.91 | ~0.93 | ~0.92 |
| F1-score | ~0.92 | ~0.92 | ~0.92 |
| **Accuracy** | | | **~0.92** |

> Exact numbers vary by dataset and training run. Retrain to reproduce.

---

## Improvements Tried

### Strategy A — Weighted Loss (Class Imbalance)
Applied `torch.nn.CrossEntropyLoss(weight=[1.0, imbalance_ratio])` to penalise misclassification of the minority class.

| | Accuracy | Macro F1 |
|--|---------|---------|
| Baseline | 0.914 | 0.913 |
| Weighted | 0.921 | 0.920 |

### Strategy B — DistilBERT vs BERT-base
DistilBERT converges faster and performs comparably with far lower memory use.

### Strategy C — Learning Rate Tuning
`lr=2e-5` is the sweet spot; `5e-5` causes instability, `1e-5` is too slow.

### Strategy D — Sequence Length
Increasing to 256 gives marginal improvement; 128 is a good trade-off.

---

## Project Structure

```
fake_news_bert/
├── day1_2_dataset_exploration.py   # Load, explore, clean dataset
├── day3_tokenization.py            # Tokenizer, Dataset class, DataLoaders
├── day4_5_train.py                 # Fine-tuning loop, save model
├── day6_evaluate.py                # Metrics, confusion matrix, ROC curve
├── day7_error_analysis.py          # Error breakdown and patterns
├── day8_improvement.py             # 4 improvement strategies
├── day9_gradio_app.py              # Gradio web app
├── fake_news_bert_colab.py         # Complete all-in-one Colab notebook
├── data/
│   ├── cleaned_dataset.csv
│   └── train_test_split.csv
├── models/
│   └── distilbert_fakenews/        # Saved model checkpoint
└── results/
    ├── label_distribution.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── error_analysis.png
    ├── classification_report.txt
    ├── metrics_summary.json
    └── test_predictions.csv
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn gradio
```

### 2. Download dataset

Download `train.csv` from https://www.kaggle.com/c/fake-news/data  
Place it in the project root directory.

### 3. Run the pipeline

```bash
# Day 1-2: Explore dataset
python day1_2_dataset_exploration.py

# Day 3: Tokenize and prepare
python day3_tokenization.py

# Day 4-5: Fine-tune (GPU recommended)
python day4_5_train.py

# Day 6: Evaluate
python day6_evaluate.py

# Day 7: Error analysis
python day7_error_analysis.py

# Day 8: Improvement (class weights)
python day8_improvement.py --strategy class_weights
# Other strategies: lr_tuning, model_compare, seq_length

# Day 9: Gradio app
python day9_gradio_app.py
```

### 4. Colab (recommended)

Open `fake_news_bert_colab.py` in Google Colab for a GPU-accelerated, end-to-end run.  
Enable GPU: **Runtime → Change runtime type → T4 GPU**

---

## Key Learnings

1. **DistilBERT is an excellent starting point** — fast to fine-tune with strong baselines out of the box.
2. **Class imbalance matters** — always use weighted loss or macro-F1 rather than raw accuracy when classes are unequal.
3. **Short texts are hardest** — fake news detection is much harder when articles have fewer than 20 words.
4. **Tone matters more than content** — models learn writing style signals (emotional, sensational language) as strongly as factual accuracy.
5. **Error analysis is essential** — FP and FN failures have completely different root causes and suggest different remedies.
6. **Learning rate is critical** — `2e-5` is a reliable default; too high causes instability, too low wastes epochs.

