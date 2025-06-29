# NLP Projects

This repository contains a collection of NLP projects completed as part of the Natural Language Processing (NLP) course at **Ben-Gurion University of the Negev**. The projects demonstrate core techniques in language modeling, word embeddings, and applied NLP systems like spell correction and text classification.

---

## Projects

### 1. Context-Sensitive Spell Checker (Noisy Channel Model)
A robust spell checker based on:
- **Trigram word-level language model**
- **Character-level confusion matrices**
- **Noisy channel model** with smoothing and error probability modeling

> 📄 File: `Word Correction.py`

Key Features:
- One-word correction per sentence using edit distance (edits1, edits2)
- Support for substitution, deletion, insertion, and transposition
- Log-likelihood evaluation with Laplace smoothing
- Can operate with character- or word-based models

---

### 2. Skip-Gram Word Embedding Model with Negative Sampling
An implementation of the **Skip-Gram model** trained on real text, capable of:
- Learning vector representations of words (embeddings)
- Computing **cosine similarity** between words
- Solving **analogy tasks** (e.g., king – man + woman ≈ queen)

> 📄 File: `Skip Grams.py`

Key Features:
- Context window control and negative sampling
- Early stopping and checkpointing
- Supports five vector combination modes (T, C, avg, sum, concat)
- Analogy tests and closest-word queries

---

### 3. Trump Tweet Classification (Zip Archive)
A text classification project focused on identifying the **author of Trump-related tweets** using NLP-based features and classification algorithms (e.g., Naive Bayes, Logistic Regression).

> 📦 File: `Trump Classification.zip`  
> *(Please unzip for full code and documentation)*

Key Features:
- Tweet preprocessing and feature extraction
- Word frequency analysis and TF-IDF
- Model training and evaluation

---

## Requirements:

- Python 3.x
- `numpy`, `pandas`, `nltk`, `pickle`
- For spell checker: requires text corpus and error confusion matrices
- For Skip-Gram: sentence tokenized input text

---

## License

This repository is for educational and demonstration purposes only.  
