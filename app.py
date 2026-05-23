import streamlit as st
import mne
import numpy as np
import torch
import torch.nn as nn
import tempfile
import plotly.graph_objects as go
from scipy.signal import welch

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="EEG Stress Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Real-Time EEG Stress Detection ")
st.write(
    " Upload EEG Signal"
    " Pre Processing & Filtering, Epoch Segmentation, Stress Percentage Calculation,displays the final stress percentage."
)

# =====================================================
# CONSTANTS
# =====================================================

MODEL_PATH = "eegnet_best.pt"

EPOCH_SEC = 2.0
STEP_SEC = 2.0

LOW_FREQ = 1.0
HIGH_FREQ = 40.0
NOTCH_FREQ = 50.0

DROP_CH_KEYWORDS = [
    "a1", "a2", "x1", "x2",
    "ecg", "ekg", "emg",
    "stim", "status"
]

# =====================================================
# EEGNET MODEL ARCHITECTURE
# =====================================================

class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes,
        n_channels,
        n_times,
        sfreq,
        F1=8,
        D=2,
        dropout=0.5
    ):
        super().__init__()

        F2 = F1 * D

        kern_temporal = int(sfreq // 2)
        pad_temporal = kern_temporal // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, kern_temporal),
                padding=(0, pad_temporal),
                bias=False
            ),
            nn.BatchNorm2d(F1),

            nn.Conv2d(
                F1,
                F2,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                F2,
                F2,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F2,
                bias=False
            ),
            nn.Conv2d(
                F2,
                F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block2(self.block1(dummy))
            flat_size = int(np.prod(out.shape[1:]))

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model = EEGNet(
        n_classes=2,
        n_channels=checkpoint["n_channels"],
        n_times=checkpoint["n_times"],
        sfreq=checkpoint["sfreq"],
        dropout=0.5
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint

# =====================================================
# CHANNEL CLEANING
# =====================================================

def auto_drop_channels(raw):
    drop_channels = []

    for ch in raw.ch_names:
        ch_lower = ch.lower()
        if any(keyword in ch_lower for keyword in DROP_CH_KEYWORDS):
            drop_channels.append(ch)

    if len(drop_channels) > 0:
        raw.drop_channels(drop_channels)

    return raw

# =====================================================
# PREPROCESSING + 2 SEC EPOCHING
# =====================================================

def preprocess_and_create_epochs(file_path, checkpoint):

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=False
    )

    raw = auto_drop_channels(raw)

    trained_sfreq = checkpoint["sfreq"]

    if raw.info["sfreq"] != trained_sfreq:
        raw.resample(trained_sfreq, verbose=False)

    raw.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        fir_design="firwin",
        verbose=False
    )

    raw.notch_filter(
        freqs=NOTCH_FREQ,
        verbose=False
    )

    data = raw.get_data()

    data = (
        data - data.mean(axis=1, keepdims=True)
    ) / (
        data.std(axis=1, keepdims=True) + 1e-8
    )

    required_channels = checkpoint["n_channels"]

    if data.shape[0] > required_channels:
        data = data[:required_channels, :]

    if data.shape[0] < required_channels:
        raise ValueError(
            f"Uploaded EEG file has {data.shape[0]} channels, "
            f"but the trained EEGNet model needs {required_channels} channels."
        )

    epoch_samples = int(EPOCH_SEC * trained_sfreq)

    epochs = []

    for start in range(
        0,
        data.shape[1] - epoch_samples + 1,
        epoch_samples
    ):
        epoch = data[:, start:start + epoch_samples]
        epochs.append(epoch)

    if len(epochs) == 0:
        raise ValueError("EEG file is too short. Minimum 2 seconds is required.")

    epochs = np.array(epochs, dtype=np.float32)

    epochs = epochs[:, np.newaxis, :, :]

    return raw, epochs

# =====================================================
# PREDICTION
# =====================================================

def predict_stress(model, epochs):

    X = torch.FloatTensor(epochs)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1).numpy()

    stress_probs = probs[:, 1]

    stress_percentage = float(np.mean(stress_probs) * 100)

    final_prediction = 1 if stress_percentage >= 50 else 0

    return stress_percentage, final_prediction, stress_probs

# =====================================================
# STRESS GAUGE VISUALIZATION
# =====================================================

def stress_gauge(stress_percentage):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=stress_percentage,
            number={"suffix": "%"},
            title={"text": "Stress Percentage"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 40], "color": "#b7f7c1"},
                    {"range": [40, 70], "color": "#fff3b0"},
                    {"range": [70, 100], "color": "#ffb3b3"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": stress_percentage
                }
            }
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# SIMPLE EEG PREVIEW
# =====================================================

def eeg_preview(raw):

    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    samples = int(2 * sfreq)
    time = np.arange(samples) / sfreq

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=data[0, :samples],
            mode="lines",
            name="First EEG Channel"
        )
    )

    fig.update_layout(
        title="2-Second EEG Signal Preview",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PSD PREVIEW
# =====================================================

def psd_preview(raw):

    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    freqs, psd = welch(
        data[0],
        fs=sfreq,
        nperseg=min(256, data.shape[1])
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=psd,
            mode="lines",
            name="PSD"
        )
    )

    fig.update_layout(
        title="Power Spectral Density Preview",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        xaxis=dict(range=[0, 50]),
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# MAIN APP
# =====================================================

try:
    model, checkpoint = load_model()
    st.success("EEGNet model loaded successfully.")
except Exception as e:
    st.error("Model loading failed. Make sure eegnet_best.pt is present in the same folder.")
    st.write(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your EEG EDF file",
    type=["edf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        with st.spinner("Processing EEG signal..."):
            raw, epochs = preprocess_and_create_epochs(file_path, checkpoint)
            stress_percentage, final_prediction, stress_probs = predict_stress(model, epochs)

        st.subheader("EEG File Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sampling Frequency", f"{raw.info['sfreq']} Hz")

        with col2:
            st.metric("Channels Used", len(raw.ch_names))

        with col3:
            st.metric("2-sec Epochs Created", epochs.shape[0])

        st.subheader("Final Stress Detection Result")

        stress_gauge(stress_percentage)

        if final_prediction == 1:
            st.error("Final Prediction: Stressed / Task State")
        else:
            st.success("Final Prediction: Relaxed / No-Stress State")

        st.write(
            f"The uploaded EEG file was divided into **{epochs.shape[0]} samples**, "
            f"where each sample is a **2-second EEG epoch**."
        )

        st.subheader("EEG Signal Preview")
        eeg_preview(raw)

        st.subheader("PSD Preview")
        psd_preview(raw)

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)

else:
    st.info("Upload an EDF EEG file to start prediction.")
