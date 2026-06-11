# 🧠 Adaptive EEG Stress Detection System

## Overview

An AI-powered biomedical application that analyzes EEG (Electroencephalography) signals to estimate stress levels from brain activity.

The system automatically adapts to different EEG channel configurations and performs real-time stress prediction using deep learning models. It supports both clinical-grade 64-channel EEG recordings and practical 19-channel EEG recordings through a unified processing pipeline.

---

## Key Highlights

* Adaptive EEG Channel Processing
* Real-Time Stress Prediction
* Automated EEG Preprocessing Pipeline
* Multi-Model Research Comparison
* Deep Learning-Based Classification
* Biomedical Signal Analysis
* Streamlit-Based Interactive Interface
* Support for 19CH and 64CH EEG Systems

---

## Project Workflow

Upload EEG (.edf)
→ Channel Detection & Adaptation
→ Artifact Reduction
→ Bandpass Filtering (1–40 Hz)
→ Notch Filtering (50 Hz)
→ Signal Normalization
→ 2-Second Epoch Segmentation
→ EEGNet Inference
→ Stress Probability Estimation
→ Final Stress Assessment

---

## System Architecture

┌──────────────────────────┐

│      EEG EDF FILE        │

│   (19CH / 64CH / Other)  │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│   EEG Data Acquisition   │

│      EDF File Reader     │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│ Channel Detection Module │

│  Detect Available EEG CH │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│ Adaptive Channel Engine  │

│ 19CH / 64CH Selection    │

│ Missing CH Handling      │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│   EEG Preprocessing      │

│ • Resampling (256 Hz)    │

│ • Artifact Removal       │

│ • Bandpass 1-40 Hz       │

│ • Notch 50 Hz            │

│ • Normalization          │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│ 2-Second Epoch Creation  │

│  Fixed Length Samples    │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│    EEGNet Inference      │

│ 19CH Model / 64CH Model  │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│ Stress Probability Calc  │

│ Epoch-wise Prediction    │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│ Stress Level Estimation  │

│ Normal / Moderate / High │

│       / Severe           │

└────────────┬─────────────┘

             │

             ▼

┌──────────────────────────┐

│ Streamlit Visualization  │

│ Stress % + Recommendations│

└──────────────────────────┘

---

## Adaptive Channel Pipeline

```
         EEG EDF FILE
                │
                ▼
   Detect Available Channels
                │
                ▼
   Remove Reference Channels
```

(A1,A2,M1,M2,EOG,ECG,EMG,REF...)
│
▼
Standardize Channel Names
(Fp1→FP1, EEG FP1-REF→FP1, etc.)
│
▼
Count EEG Channels
│
┌───────────┼───────────┐
│                       │
▼                       ▼
Channels ≥ 64          19 ≤ Channels < 64
│                       │
▼                       ▼
Select 64CH Model      Select 19CH Model
│                       │
└───────────┬───────────┘
│
▼
Check Required Channels
│
┌─────────────┴─────────────┐
│                           │
▼                           ▼
All Channels Available    Missing Channels
│                           │
▼                           ▼
Normal Processing       Use Available Channels
│                  Adaptive Mapping
│                  Missing CH Ignored
▼                           ▼
└─────────────┬─────────────┘
│
▼
EEG Preprocessing
│
▼
EEGNet Prediction
│
▼
Stress Percentage

---

## Model Performance Comparison

| Model       | Dataset       | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| ----------- | ------------- | ------------ | ------------- | ---------- | ------------ |
| CNN         | Live EEG      | 50.0         | 56.0          | 50.0       | 34.0         |
| CNN + SVM   | Live EEG      | 71.2         | 67.0          | 71.0       | 69.0         |
| EEGNet 19CH | Live EEG      | 60.5         | 61.0          | 61.0       | 60.0         |
| EEGNet 64CH | PhysioNet EEG | 80.7         | 80.0          | 81.0       | 80.0         |

Detailed results are available in:

`results/model_comparison.csv`

---

## Resource 

* Deployed EEGNet Models
* Comparative Research Models 
* Training Notebooks 
* Dataset Documentation 
* Streamlit Application
* System Architecture Diagrams 
* Model Performance Results

---

## Technologies Used

### Artificial Intelligence

* PyTorch
* EEGNet
* CNN
* SVM
* Random Forest

### Biomedical Signal Processing

* MNE-Python
* NumPy
* SciPy

---

## Datasets

### Live EEG Dataset

Custom-recorded 19-channel EEG recordings used for model development and comparative analysis.

### PhysioNet EEG Dataset

Public EEG dataset used for training and validating the 64-channel EEGNet model.

---

## Research Contribution

This project investigates the impact of different machine learning and deep learning approaches for EEG-based stress detection while introducing an adaptive channel-processing framework capable of handling multiple EEG acquisition setups through a unified inference pipeline.

---

## Demonstration

### Stress Detection Example

### Streamlit Interface

https://eeg-stress-detection-app-7hcgdqezjoyiqoyesgmwzv.streamlit.app/#model-used-19ch-eeg-net

---

## Future Improvements

* Subject-Independent Stress Classification
* Cognitive State Analysis
* Attention and Mental Workload Monitoring
* Cross-Dataset Generalization
* Edge AI Deployment for Portable EEG Systems

---

## Author

**Chireddy Madhava Reddy**

Email:madhavareddy.official@gmail.com
