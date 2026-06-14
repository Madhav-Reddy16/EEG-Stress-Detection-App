# 🧠 AI-Driven Biomedical Stress Detection using Adaptive EEG Channel Processing and Deep Learning

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

========================================================================
                          SYSTEM ARCHITECTURE                           
========================================================================
 ```
+------------------------------------------+
|         EEG EDF FILE INPUT               |
|       (19CH / 64CH / Other Data)         |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|        EEG DATA ACQUISITION              |
|           EDF File Reader                |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|      CHANNEL DETECTION MODULE            |
|    Detects Available EEG Channels        |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|       ADAPTIVE CHANNEL ENGINE            |
|  * Selects 19CH / 64CH Model             |
|  * Handles Missing Channels              |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|          EEG PREPROCESSING               |
|  * Resampling (Target: 256 Hz)           |
|  * Artifact Removal                      |
|  * Bandpass Filtering (1 - 40 Hz)        |
|  * Notch Filtering (50 Hz)               |
|  * Amplitude Normalisation               |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|       2-SECOND EPOCH CREATION            |
|     Fixed-Length Segment Samples         |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|          EEGNET INFERENCE                |
|      19CH / 64CH Model Prediction        |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|     STRESS PROBABILITY ANALYSIS          |
|        Epoch-wise Predictions            |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|       STRESS LEVEL ESTIMATION            |
|  Normal / Moderate / High / Severe       |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
|       STREAMLIT VISUALISATION            |
|    Stress % & Recommendations            |
+------------------------------------------+




=========================================================================
                        ADAPTIVE CHANNEL PIPELINE                        
=========================================================================

┌─────────────────────────────────────────────────────────┐
│                      EEG EDF FILE                       │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                Detect Available Channels                │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                Remove Reference Channels                │
│           (A1,A2,M1,M2,EOG,ECG,EMG,REF...)              │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                Standardize Channel Names                │
│            (Fp1→FP1, EEG FP1-REF→FP1, etc.)             │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   Count EEG Channels                    │
└────────────────────────────┬────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
 ┌───────────────────────┐       ┌───────────────────────┐
 │     Channels ≥ 64     │       │  19 ≤ Channels < 64   │
 └───────────┬───────────┘       └───────────┬───────────┘
             │                               │
             ▼                               ▼
 ┌───────────────────────┐       ┌───────────────────────┐
 │   Select 64CH Model   │       │   Select 19CH Model   │
 └───────────┬───────────┘       └───────────┬───────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 Check Required Channels                 │
└────────────────────────────┬────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
 ┌───────────────────────┐       ┌───────────────────────┐
 │All Channels Available │       │   Missing Channels    │
 │   Normal Processing   │       │Use Available Channels │
 │                       │       │   Adaptive Mapping    │
 │                       │       │  Missing CH Ignored   │
 └───────────┬───────────┘       └───────────┬───────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    EEG Preprocessing                    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    EEGNet Prediction                    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    Stress Percentage                    │
└─────────────────────────────────────────────────────────┘



## Model Performance Comparison

| Model         | Dataset       | Dataset Size    | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |                           
| ------------- | ------------- | --------------- | ------------ | ------------- | ---------- | ------------ |  
| CNN           | Live EEG      | 125 Samples     | 50.0         | 56.0          | 50.0       | 34.0         |            
| CNN + SVM     | Live EEG      | 125 Samples     | 71.2         | 67.0          | 71.0       | 69.0         |                
| Random Forest | Live EEG      | 32 Test Samples | 78.1         | 71.4          | **93.8**   | 81.1         |       
| EEGNet 19CH   | Live EEG      | 250 Epochs      | 60.5         | 61.0          | 61.0       | 60.0         |           
| EEGNet 64CH   | PhysioNet EEG | 109 Subjects    | **80.7**     | **80.0**      | **81.0**   | **80.0**     |           


Detailed results are available in:

Results

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
