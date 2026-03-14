# Mini Project 9: Content Moderation with Transformers
**COMP 9130 — Applied Artificial Intelligence**

## Team Members
- [Your Name]
- [Partner Name]

---

## Problem Description

SafeSpace AI needs an automated content moderation system that can classify social media posts into three categories:
1. **Hate speech** — content attacking individuals/groups based on protected characteristics
2. **Offensive language** — rude/vulgar content that doesn't target protected groups
3. **Neither** — acceptable content

The critical challenge is distinguishing hate speech from merely offensive language, as this distinction has legal and regulatory implications.

---

## Dataset

**Source:** [Twitter Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)  
**Citation:** Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017)

**Statistics:**
- Total samples: ~25,000 tweets
- Classes: 3 (hate speech: 0, offensive: 1, neither: 2)
- **Severe class imbalance**: ~5% hate, ~77% offensive, ~18% neither
- Average length: 100-140 characters

**Challenges:**
- Class imbalance (handled with weighted loss)
- Noisy text (slang, abbreviations, misspellings)
- Subtle distinction between hate speech and offensive language
- Context-dependent meaning

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone [your-repo-url]
cd mini-project-9
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or in Google Colab:
```python
!pip install transformers torch scikit-learn pandas numpy matplotlib seaborn tqdm
```

### 3. Run Notebooks (in order)

**Notebook 1: Data Exploration**
```
notebooks/01_exploration.ipynb
```
- Loads and explores dataset
- Analyzes class distribution
- Cleans text
- Creates train/val/test splits

**Notebook 2: TF-IDF Baseline**
```
notebooks/02_baseline.ipynb
```
- Builds TF-IDF + Logistic Regression baseline
- Tunes hyperparameters
- Evaluates performance

**Notebook 3: Transformer Fine-Tuning**
```
notebooks/03_transformer.ipynb
```
- Fine-tunes DistilBERT for 3-class classification
- Handles class imbalance with weighted loss
- Compares with baseline
- Analyzes errors and confidence

### 4. Hardware Requirements

- **CPU**: Works but slow (~30 min/epoch for transformer)
- **GPU** (recommended): Free Google Colab T4 GPU (~5 min/epoch)
  - In Colab: Runtime → Change runtime type → T4 GPU

---

## Results Summary

### Model Comparison

| Metric | TF-IDF Baseline | DistilBERT | Improvement |
|--------|-----------------|------------|-------------|
| **Accuracy** | [X.XX] | [X.XX] | [+X.X%] |
| **F1 (macro)** | [X.XX] | [X.XX] | [+X.X%] |
| **F1 (weighted)** | [X.XX] | [X.XX] | [+X.X%] |

### Per-Class Performance (DistilBERT)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Hate speech | [X.XX] | [X.XX] | [X.XX] | [XXX] |
| Offensive | [X.XX] | [X.XX] | [X.XX] | [XXXX] |
| Neither | [X.XX] | [X.XX] | [X.XX] | [XXX] |

### Key Findings

**Strengths:**
- [e.g., DistilBERT significantly outperforms TF-IDF on minority class (hate speech)]
- [e.g., Handles slang and misspellings better than TF-IDF]

**Weaknesses:**
- [e.g., Still confuses hate speech with offensive language in X% of cases]
- [e.g., Struggles with sarcasm and context-dependent meaning]

**Production Recommendations:**
- Confidence threshold: [X.XX] for auto-moderation
- Expected human review: [X%] of posts
- Priority: Minimize false negatives (missing hate speech) over false positives

---

## Project Structure

```
mini-project-9/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── notebooks/
│   ├── 01_exploration.ipynb      # Data exploration & preprocessing
│   ├── 02_baseline.ipynb         # TF-IDF baseline model
│   └── 03_transformer.ipynb      # DistilBERT fine-tuning
├── data/
│   ├── train.csv                 # Training split
│   ├── val.csv                   # Validation split
│   ├── test.csv                  # Test split
│   ├── baseline_results.json     # TF-IDF results
│   ├── transformer_results.json  # DistilBERT results
│   ├── best_model.pt            # Saved DistilBERT weights
│   └── *.png                     # Generated plots
└── report.pdf                     # Final report (upload to Learning Hub)
```

---

## Methodology

### 1. Data Preprocessing
- Basic cleaning for transformers (remove URLs, keep @mentions/hashtags)
- Aggressive cleaning for TF-IDF (lowercase, remove special characters)
- Stratified train/val/test split (64%/16%/20%)

### 2. TF-IDF Baseline
- Vectorization: 5000 features, unigrams + bigrams
- Model: Logistic Regression with balanced class weights
- Hyperparameter tuning: GridSearchCV on regularization strength (C)

### 3. Transformer Fine-Tuning
- Model: DistilBERT-base-uncased (66M parameters)
- Tokenization: Max length 128 tokens
- **Class imbalance handling**: Weighted CrossEntropyLoss
- Hyperparameters:
  - Learning rate: 3e-5
  - Batch size: 16
  - Epochs: 4
  - Optimizer: AdamW with linear warmup
- Training: Manual PyTorch loop with gradient clipping

---

## Error Analysis

### Common Failure Patterns

1. **Sarcasm/Irony** ([X]% of errors)
   - Example: [quote a misclassified tweet]
   
2. **Context-dependent language** ([X]% of errors)
   - Example: [quote a misclassified tweet]

3. **Hate vs Offensive confusion** ([X]% of errors)
   - Example: [quote a misclassified tweet]

4. **Annotation disagreement** ([X]% of errors)
   - Example: [tweets that seem mislabeled in ground truth]

---

## Production Workflow Design

### Confidence Thresholds
- **Auto-removal** (hate speech): Confidence ≥ [X.XX]
- **Auto-approval** (neither): Confidence ≥ [X.XX]
- **Human review**: Confidence < [X.XX] or borderline cases

### Scalability
At 100K posts/day:
- **Auto-moderated**: [X]% = [XX,XXX] posts/day
- **Human review**: [X]% = [X,XXX] posts/day

### Error Cost Analysis
False negatives (missing hate speech) are MORE costly because:
- Legal/regulatory penalties
- User harm and platform reputation damage
- Potential for viral spread of harmful content

Therefore:
- Use lower threshold for hate speech detection
- Accept higher false positive rate
- Route borderline cases to human moderators

---

## Limitations & Future Work

### Current Limitations
1. **Class imbalance**: Only ~5% hate speech in training data
2. **Single language**: English-only dataset
3. **Static model**: Doesn't adapt to evolving slang/terminology
4. **No user context**: Doesn't consider user history or social graph

### Version 2 Recommendations
1. **Data collection**: Actively collect more hate speech examples
2. **Multi-lingual support**: Expand to non-English content
3. **Active learning**: Flag low-confidence predictions for human labeling
4. **Contextual features**: Incorporate user history, engagement patterns
5. **Ensemble approach**: Combine transformer with rule-based filters
6. **Regular retraining**: Monthly updates with new examples

---

## References

1. Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language*. Proceedings of ICWSM.

2. Hugging Face. (2023). *Fine-tuning a Pretrained Model*. https://huggingface.co/docs/transformers/training

3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT*. arXiv preprint arXiv:1910.01108.

---

## Team Contributions

**[Your Name]:**
- [Your specific contributions]

**[Partner Name]:**
- [Partner's specific contributions]

**Collaborative work:**
- [What you did together]

---

## Acknowledgments

- COMP 9130 course materials and in-class activities (Week 10)
- Hugging Face Transformers library and documentation
- Davidson et al. for the hate speech dataset

---

**Contact:** [your emails]  
**Submission Date:** [date]
