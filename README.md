# MemoTag Speech Intelligence PoC

This repository contains a proof-of-concept (PoC) for detecting early cognitive decline patterns from voice data, developed for MemoTag's speech intelligence module. The pipeline processes simulated audio clips, extracts clinically relevant features, applies unsupervised machine learning, and identifies abnormal samples indicative of cognitive impairment. It also supports real-time prediction for user-uploaded audio.

## Overview
The PoC uses 8 simulated audio clips (4 healthy, 4 impaired) and allows users to upload audio for instant analysis:
- Preprocesses audio (normalization, MP3-to-WAV conversion, transcription).
- Extracts features like pauses, hesitations, speech rate, and incomplete sentences.
- Applies K-Means clustering and Isolation Forest for anomaly detection.
- ## Predicts cognitive decline risk for uploaded audio in real-time.
- Generates visualizations and a detailed report.

## Methodology
- **Data**: Simulated clips mimic healthy and impaired speech (hesitations, vague terms) to respect privacy. User-uploaded audio is supported for real-time testing.
- **Preprocessing**: Audio normalized to 16kHz, transcribed using Google Speech-to-Text.
- **Features**:
  - Pauses per sentence (>300ms).
  - Hesitation markers (“uh,” “um”).
  - Word recall issues (repetitions, vague terms like “thing”).
  - Speech rate (words/sec).
  - Pitch variability.
  - Naming issues and incomplete sentences.
- **Machine Learning**:
  - **K-Means**: Clusters samples into healthy vs. impaired.
  - **Isolation Forest**: Flags outliers (contamination=0.25).
  - **Real-Time Prediction**: Classifies user audio using trained models.

## Findings
- **Key Features**:
  - Hesitation ratio: Higher in impaired samples (0.1–0.2 vs. 0).
  - Speech rate: Slower in impaired samples (1–2 vs. 3–4 words/sec).
  - Incomplete sentences: 20–40% in impaired samples.
- **Results**:
  - Clustering separated healthy (Cluster 0) and impaired (Cluster 1).
  - Anomaly detection flagged 3/4 impaired samples.
  - Real-time prediction enables instant analysis of user audio.
- **Visualizations**: Scatter plots, heatmaps, and feature distributions (see `reports/`).

## Next Steps
- Validate with real datasets (e.g., ADReSS).
- Add prosodic features (jitter, shimmer) and semantic analysis (BERT).
- Explore supervised learning with labeled data.
- Enhance real-time prediction with microphone input (outside Colab).

## Repository Structure
- `MemoTag_Speech_PoC.ipynb`: Main Jupyter notebook with pipeline and real-time prediction.
- `data/processed/`:
  - `metadata.csv`: Audio metadata and transcriptions.
  - `features.csv`: Extracted features.
  - `results.csv`: Clustering and anomaly results.
  - `*.wav`: Processed audio files.
- `reports/`:
  - `MemoTag_Report.md`: Detailed report.
  - `clusters.png`, `corr_heatmap.png`, `feature_distributions.png`: Visualizations.

## Running the Notebook
1. Open in Google Colab: [Upload `MemoTag_Speech_PoC.ipynb` to Colab](https://colab.research.google.com/) or use the GitHub URL.
2. Install dependencies (included in notebook):
   ```bash
   !pip install librosa soundfile speechrecognition gtts spacy scikit-learn matplotlib seaborn pydub
   !apt-get install -y ffmpeg
   !python -m spacy download en_core_web_sm
