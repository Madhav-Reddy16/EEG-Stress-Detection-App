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
 
<img width="1440" height="1564" alt="image" src="https://github.com/user-attachments/assets/017ecfc5-328f-4448-9079-c1ae8d275837" /><img width="1440" height="1564" alt="image" src="https://github.com/user-attachments/assets/dc2cbb3b-194e-4d30-859b-45fc6dfd6256" />


=========================================================================
                        ADAPTIVE CHANNEL PIPELINE                        
=========================================================================

<svg width="700" height="1080" viewBox="0 0 700 1080" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#666" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <rect width="700" height="1080" fill="#0d1117"/>

  <!-- Node 1: EEG EDF File Input -->
  <rect x="150" y="20" width="400" height="60" rx="12" fill="#d4f5e9" stroke="#6fcfa5" stroke-width="1.2"/>
  <text x="350" y="45" text-anchor="middle" font-size="15" font-weight="700" fill="#1a5c3a">EEG EDF File Input</text>
  <text x="350" y="65" text-anchor="middle" font-size="12" fill="#2e7d52">19CH / 64CH / Other Data</text>
  <line x1="350" y1="80" x2="350" y2="104" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 2: Detect Available Channels -->
  <rect x="150" y="104" width="400" height="60" rx="12" fill="#d4f5e9" stroke="#6fcfa5" stroke-width="1.2"/>
  <text x="350" y="129" text-anchor="middle" font-size="15" font-weight="700" fill="#1a5c3a">Detect Available Channels</text>
  <text x="350" y="149" text-anchor="middle" font-size="12" fill="#2e7d52">Scans all channels in EDF file</text>
  <line x1="350" y1="164" x2="350" y2="188" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 3: Remove Reference Channels -->
  <rect x="150" y="188" width="400" height="60" rx="12" fill="#d4f5e9" stroke="#6fcfa5" stroke-width="1.2"/>
  <text x="350" y="213" text-anchor="middle" font-size="15" font-weight="700" fill="#1a5c3a">Remove Reference Channels</text>
  <text x="350" y="233" text-anchor="middle" font-size="12" fill="#2e7d52">A1, A2, M1, M2, EOG, ECG, EMG, REF...</text>
  <line x1="350" y1="248" x2="350" y2="272" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 4: Standardize Channel Names -->
  <rect x="150" y="272" width="400" height="60" rx="12" fill="#d4f5e9" stroke="#6fcfa5" stroke-width="1.2"/>
  <text x="350" y="297" text-anchor="middle" font-size="15" font-weight="700" fill="#1a5c3a">Standardize Channel Names</text>
  <text x="350" y="317" text-anchor="middle" font-size="12" fill="#2e7d52">Fp1&#x2192;FP1, EEG FP1-REF&#x2192;FP1, etc.</text>
  <line x1="350" y1="332" x2="350" y2="356" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 5: Count EEG Channels -->
  <rect x="150" y="356" width="400" height="60" rx="12" fill="#d4f5e9" stroke="#6fcfa5" stroke-width="1.2"/>
  <text x="350" y="381" text-anchor="middle" font-size="15" font-weight="700" fill="#1a5c3a">Count EEG Channels</text>
  <text x="350" y="401" text-anchor="middle" font-size="12" fill="#2e7d52">Total clean EEG channels found</text>

  <!-- Branch split from Count -->
  <line x1="350" y1="416" x2="350" y2="430" stroke="#666" stroke-width="1.5"/>
  <line x1="200" y1="430" x2="500" y2="430" stroke="#666" stroke-width="1.5"/>
  <line x1="200" y1="430" x2="200" y2="450" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="500" y1="430" x2="500" y2="450" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 6a: Channels >= 64 -->
  <rect x="75" y="450" width="250" height="56" rx="10" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="200" y="474" text-anchor="middle" font-size="13" font-weight="700" fill="#3b2a8a">Channels &#x2265; 64</text>
  <text x="200" y="493" text-anchor="middle" font-size="11" fill="#5a489c">High-density EEG</text>

  <!-- Node 6b: 19 <= Channels < 64 -->
  <rect x="375" y="450" width="250" height="56" rx="10" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="500" y="474" text-anchor="middle" font-size="13" font-weight="700" fill="#3b2a8a">19 &#x2264; Channels &lt; 64</text>
  <text x="500" y="493" text-anchor="middle" font-size="11" fill="#5a489c">Standard-density EEG</text>

  <line x1="200" y1="506" x2="200" y2="526" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="500" y1="506" x2="500" y2="526" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 7a: Select 64CH Model -->
  <rect x="75" y="526" width="250" height="56" rx="10" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="200" y="550" text-anchor="middle" font-size="13" font-weight="700" fill="#3b2a8a">Select 64CH Model</text>
  <text x="200" y="569" text-anchor="middle" font-size="11" fill="#5a489c">eegnet_64ch.pt</text>

  <!-- Node 7b: Select 19CH Model -->
  <rect x="375" y="526" width="250" height="56" rx="10" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="500" y="550" text-anchor="middle" font-size="13" font-weight="700" fill="#3b2a8a">Select 19CH Model</text>
  <text x="500" y="569" text-anchor="middle" font-size="11" fill="#5a489c">eegnet_19ch.pt</text>

  <!-- Merge after model selection -->
  <line x1="200" y1="582" x2="200" y2="602" stroke="#666" stroke-width="1.5"/>
  <line x1="500" y1="582" x2="500" y2="602" stroke="#666" stroke-width="1.5"/>
  <line x1="200" y1="602" x2="500" y2="602" stroke="#666" stroke-width="1.5"/>
  <line x1="350" y1="602" x2="350" y2="622" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 8: Check Required Channels -->
  <rect x="150" y="622" width="400" height="60" rx="12" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="350" y="647" text-anchor="middle" font-size="15" font-weight="700" fill="#3b2a8a">Check Required Channels</text>
  <text x="350" y="667" text-anchor="middle" font-size="12" fill="#5a489c">Validate against model channel list</text>

  <!-- Branch split from Check -->
  <line x1="350" y1="682" x2="350" y2="696" stroke="#666" stroke-width="1.5"/>
  <line x1="200" y1="696" x2="500" y2="696" stroke="#666" stroke-width="1.5"/>
  <line x1="200" y1="696" x2="200" y2="716" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="500" y1="696" x2="500" y2="716" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 9a: All Channels Available -->
  <rect x="75" y="716" width="250" height="76" rx="10" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="200" y="738" text-anchor="middle" font-size="13" font-weight="700" fill="#3b2a8a">All Channels Available</text>
  <text x="200" y="756" text-anchor="middle" font-size="11" fill="#5a489c">Normal Processing</text>
  <text x="200" y="774" text-anchor="middle" font-size="11" fill="#5a489c">Full model input</text>

  <!-- Node 9b: Missing Channels -->
  <rect x="375" y="716" width="250" height="76" rx="10" fill="#e8e4f7" stroke="#9b8fd4" stroke-width="1.2"/>
  <text x="500" y="733" text-anchor="middle" font-size="13" font-weight="700" fill="#3b2a8a">Missing Channels</text>
  <text x="500" y="750" text-anchor="middle" font-size="11" fill="#5a489c">Use Available Channels</text>
  <text x="500" y="767" text-anchor="middle" font-size="11" fill="#5a489c">Adaptive Mapping</text>
  <text x="500" y="784" text-anchor="middle" font-size="11" fill="#5a489c">Missing CH = Ignored</text>

  <!-- Merge after channel check -->
  <line x1="200" y1="792" x2="200" y2="812" stroke="#666" stroke-width="1.5"/>
  <line x1="500" y1="792" x2="500" y2="812" stroke="#666" stroke-width="1.5"/>
  <line x1="200" y1="812" x2="500" y2="812" stroke="#666" stroke-width="1.5"/>
  <line x1="350" y1="812" x2="350" y2="832" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 10: EEG Preprocessing -->
  <rect x="150" y="832" width="400" height="76" rx="12" fill="#dce8fb" stroke="#7aaee8" stroke-width="1.2"/>
  <text x="350" y="856" text-anchor="middle" font-size="15" font-weight="700" fill="#1a3d6b">EEG Preprocessing</text>
  <text x="350" y="876" text-anchor="middle" font-size="12" fill="#2a5a9c">Resampling (256 Hz) &#xB7; Artifact Removal</text>
  <text x="350" y="894" text-anchor="middle" font-size="12" fill="#2a5a9c">Bandpass 1&#x2013;40 Hz &#xB7; Notch 50 Hz &#xB7; Normalisation</text>
  <line x1="350" y1="908" x2="350" y2="932" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 11: EEGNet Prediction -->
  <rect x="150" y="932" width="400" height="60" rx="12" fill="#fef3e2" stroke="#f0b05a" stroke-width="1.2"/>
  <text x="350" y="957" text-anchor="middle" font-size="15" font-weight="700" fill="#7a3e00">EEGNet Prediction</text>
  <text x="350" y="977" text-anchor="middle" font-size="12" fill="#a05a10">19CH / 64CH Model Inference</text>
  <line x1="350" y1="992" x2="350" y2="1012" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Node 12: Stress Percentage -->
  <rect x="150" y="1012" width="400" height="60" rx="12" fill="#fde8e8" stroke="#e07070" stroke-width="1.2"/>
  <text x="350" y="1037" text-anchor="middle" font-size="15" font-weight="700" fill="#7a1a1a">Stress Percentage Output</text>
  <text x="350" y="1057" text-anchor="middle" font-size="12" fill="#a03030">Normal &#xB7; Moderate &#xB7; High &#xB7; Severe</text>

</svg>


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
