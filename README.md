# 🧠 EEG-Based Stress Detection Web Application using EEGNet and Streamlit

## 📌 Project Overview

This project is an AI-based EEG Stress Detection Web Application developed using **EEGNet Deep Learning Architecture** and **Streamlit**.

The system accepts a long-duration EEG `.edf` file, preprocesses the EEG signal, divides the signal into 2-second epochs, predicts stress for each epoch using EEGNet, and finally calculates the overall stress percentage.

The application provides an interactive web interface for EEG stress analysis and visualization.

---

# 🚀 Project Workflow

```text
Upload EEG EDF File
        ↓
Read EEG Signal using MNE
        ↓
Remove Unwanted Channels
        ↓
Bandpass Filtering (1–40 Hz)
        ↓
Notch Filtering (50 Hz)
        ↓
Signal Normalization
        ↓
2-Second Epoch Segmentation
        ↓
Each Epoch = One EEG Sample
        ↓
EEGNet Deep Learning Prediction
        ↓
Stress Probability Calculation
        ↓
Average Stress Percentage
        ↓
Final Stress Detection Result
```

---

# 🧠 Model Used

## EEGNet Deep Learning Model

EEGNet is a lightweight Convolutional Neural Network specially designed for EEG signal classification.

### Why EEGNet?

- Efficient for biomedical EEG signals
- Works well with small EEG datasets
- Low computational complexity
- Suitable for real-time applications
- Achieved high stress classification accuracy

---

# 📂 Input EEG File

The application accepts:

```text
.edf EEG recording files
```

Example:

```text
subject01.edf
```

---

# ⚙️ EEG Preprocessing Pipeline

The uploaded EEG signal undergoes the following preprocessing steps:

## 1. Channel Cleaning

Unwanted channels are removed automatically:

- ECG
- EMG
- EKG
- Stimulus channels
- Status channels

---

## 2. Bandpass Filtering

Frequency range used:

```text
1 Hz – 40 Hz
```

Purpose:

- Remove baseline drift
- Remove high-frequency noise

---

## 3. Notch Filtering

Frequency removed:

```text
50 Hz
```

Purpose:

- Remove powerline interference noise

---

## 4. Signal Normalization

Each EEG channel is normalized using:

```text
Z-score normalization
```

---

# 🧩 Epoch Segmentation

The long EEG recording is divided into small samples.

## Epoch Configuration

```text
Window Size = 2 Seconds
Step Size = 2 Seconds
```

## Non-Overlapping Epochs

Example:

```text
0–2 sec   → Epoch 1
2–4 sec   → Epoch 2
4–6 sec   → Epoch 3
```

Each epoch becomes one EEG sample for EEGNet prediction.

---

# 📊 Stress Prediction

Each epoch is passed into the trained EEGNet model.

The model predicts:

```text
Stress Probability
```

Final stress percentage is calculated by averaging all epoch predictions.

---

# 📈 App Interface

The web application displays:

- EEG File Summary
- Number of EEG Channels
- Sampling Frequency
- Number of Generated Epochs
- Stress Percentage Gauge
- Final Prediction
- EEG Signal Preview
- PSD Visualization

---

# 🖥️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Streamlit | Web Application |
| PyTorch | EEGNet Deep Learning |
| MNE | EEG Signal Processing |
| NumPy | Numerical Computation |
| SciPy | Signal Processing |
| Plotly | Interactive Visualization |

---

# 📁 Project Structure

```text
EEG-Stress-Detection-App/
│
├── app.py
├── eegnet_best.pt
├── requirements.txt
└── README.md
```

---

# 📦 Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/EEG-Stress-Detection-App.git
```

---

## Step 2: Move into Project Folder

```bash
cd EEG-Stress-Detection-App
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# ☁️ Streamlit Cloud Deployment

The project can be deployed free using Streamlit Community Cloud.

## Deployment Steps

1. Upload project files to GitHub
2. Login to Streamlit Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Click Deploy

---

# 📌 Key Features

✅ EEGNet Deep Learning Model  
✅ Automated EEG Preprocessing  
✅ 2-Second EEG Epoching  
✅ Real-Time Stress Prediction  
✅ Interactive Web Interface  
✅ Stress Percentage Visualization  
✅ PSD Signal Analysis  
✅ Free Cloud Deployment  

---

# 📊 Final Output

The system predicts:

```text
Stress Percentage (%)
```

Final result categories:

- Relaxed / No-Stress State
- Stressed / Task State

---

# 🔬 Future Improvements

- Real-time EEG Streaming
- Multi-class Emotion Detection
- Attention Monitoring
- Anxiety Classification
- Brain-Computer Interface Integration
- Mobile Application Deployment

---

# 👨‍💻 Author

Developed by:

```
Madhava Reddy
```

---

# 📄 License

This project is developed for educational and research purposes.