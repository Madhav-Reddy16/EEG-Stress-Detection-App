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

=
                                    SYSTEM ARCHITECTURE                          
=
 
<img width="1440" height="1564" alt="image" src="https://github.com/user-attachments/assets/017ecfc5-328f-4448-9079-c1ae8d275837" />

=
                                   ADAPTIVE CHANNEL PIPELINE                        
=

<img width="1440" height="2680" alt="image" src="https://github.com/user-attachments/assets/61e7c07f-7127-4ce5-81e5-9defdeb5e198" />


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

<img width="1440" height="1514" alt="image" src="https://github.com/user-attachments/assets/0f2243a1-3ab0-4bfb-a6f3-bc58fc60a058" />

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
