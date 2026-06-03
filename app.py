import json
import tempfile
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import mne

TARGET_SFREQ = 128
EPOCH_SEC = 2
EPOCH_SAMPLES = 256

MODEL_19 = "eegnet_19ch.pt"
MODEL_64 = "eegnet_64ch.pt"

CHANNELS_19_JSON = "training_channels_19.json"
CHANNELS_64_JSON = "training_channels_64.json"


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, f1=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * 2, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * 2, f1 * 2, kernel_size=(1, 16), padding=(0, 8), groups=f1 * 2, bias=False),
            nn.Conv2d(f1 * 2, f1 * 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f1 * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def clean_channel_name(ch):
    return (
        ch.replace("EEG ", "")
        .replace("-REF", "")
        .replace("-LE", "")
        .replace(".", "")
        .strip()
    )


def load_channels(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
        n_channels = checkpoint.get("n_channels", None)
        n_times = checkpoint.get("n_times", 256)
    else:
        raise ValueError("Invalid model file format.")

    if n_channels is None:
        if "64" in model_path:
            n_channels = 64
        else:
            n_channels = 19

    f1 = state["block1.0.weight"].shape[0]

    model = EEGNet(n_channels=n_channels, n_times=n_times, f1=f1)
    model.load_state_dict(state)
    model.eval()

    return model


def stress_level(stress_percent):
    if stress_percent <= 30:
        return "Normal"
    elif stress_percent <= 60:
        return "Moderate Stress"
    elif stress_percent <= 80:
        return "High Stress"
    else:
        return "Severe Stress"


def get_precautions(level):
    if level == "Normal":
        return {
            "emoji": "🟢",
            "title": "NORMAL",
            "subtitle": "Your EEG activity indicates a relaxed mental state.",
            "tips": [
                "Maintain your current routine.",
                "Stay hydrated throughout the day.",
                "Continue regular physical activity.",
                "Take short breaks during long study/work sessions."
            ],
            "bg": "#d4edda",
            "border": "#28a745"
        }

    elif level == "Moderate Stress":
        return {
            "emoji": "🟡",
            "title": "MODERATE STRESS",
            "subtitle": "Mild stress-related EEG activity was detected.",
            "tips": [
                "Practice deep breathing for 5 minutes.",
                "Take a short walk or relaxation break.",
                "Reduce continuous screen exposure.",
                "Maintain 7–8 hours of sleep."
            ],
            "bg": "#fff3cd",
            "border": "#ffc107"
        }

    elif level == "High Stress":
        return {
            "emoji": "🟠",
            "title": "HIGH STRESS",
            "subtitle": "Elevated stress-related EEG activity was detected.",
            "tips": [
                "Take 15–20 minutes of rest.",
                "Avoid multitasking for some time.",
                "Drink water and relax your body.",
                "Reduce caffeine intake.",
                "Try slow breathing or meditation."
            ],
            "bg": "#ffe0b2",
            "border": "#ff9800"
        }

    else:
        return {
            "emoji": "🔴",
            "title": "SEVERE STRESS",
            "subtitle": "Significant stress-related EEG activity was detected.",
            "tips": [
                "Take immediate rest.",
                "Avoid mentally demanding tasks.",
                "Practice guided meditation or breathing.",
                "Improve sleep quality.",
                "Consult a healthcare professional if stress persists."
            ],
            "bg": "#f8d7da",
            "border": "#dc3545"
        }


def show_result_card(level):
    info = get_precautions(level)
    tips_html = "".join([f"<li>{tip}</li>" for tip in info["tips"]])

    st.markdown(
        f"""
        <div style="
            background:{info['bg']};
            border-left:10px solid {info['border']};
            padding:30px;
            border-radius:20px;
            box-shadow:0px 6px 18px rgba(0,0,0,0.18);
            margin-top:20px;
        ">
            <h1 style="text-align:center; font-size:42px;">
                {info['emoji']} {info['title']}
            </h1>
            <h3 style="text-align:center; font-weight:500;">
                {info['subtitle']}
            </h3>
            <hr>
            <h2>Precautions</h2>
            <ul style="font-size:20px; line-height:1.8;">
                {tips_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def select_model_and_channels(raw_channels, ch19, ch64):
    available = [clean_channel_name(ch) for ch in raw_channels]

    has_64 = all(ch in available for ch in ch64)
    has_19 = all(ch in available for ch in ch19)

    if len(available) >= 64 and has_64:
        return MODEL_64, ch64, "64-channel EEGNet"

    elif len(available) >= 19 and has_19:
        return MODEL_19, ch19, "19-channel EEGNet"

    else:
        missing_19 = [ch for ch in ch19 if ch not in available]
        missing_64 = [ch for ch in ch64 if ch not in available]

        raise ValueError(
            "Incompatible EEG file. Required trained channels are missing.\n\n"
            f"Missing 19ch channels: {missing_19}\n\n"
            f"Missing 64ch channels: {missing_64}"
        )


def preprocess_edf(uploaded_file, required_channels):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    rename_dict = {ch: clean_channel_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_dict)

    unwanted_keywords = [
        "EOG", "ECG", "EMG", "STI", "STIM",
        "STATUS", "MARKER", "A2-A1"
    ]

    drop_channels = [
        ch for ch in raw.ch_names
        if any(key.lower() in ch.lower() for key in unwanted_keywords)
    ]

    if drop_channels:
        raw.drop_channels(drop_channels)

    raw.pick_channels(required_channels, ordered=True)

    raw.resample(TARGET_SFREQ)
    raw.filter(1, 40, verbose=False)
    raw.notch_filter(50, verbose=False)

    data = raw.get_data()

    if data.shape[1] < EPOCH_SAMPLES:
        raise ValueError("Recording is too short. Minimum 2 seconds required.")

    n_epochs = data.shape[1] // EPOCH_SAMPLES

    clean_epochs = []
    rejected_epochs = 0

    for i in range(n_epochs):
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES

        epoch = data[:, start:end]

        if epoch.shape[1] != EPOCH_SAMPLES:
            rejected_epochs += 1
            continue

        if np.max(np.abs(epoch)) > 300e-6:
            rejected_epochs += 1
            continue

        mean = epoch.mean(axis=1, keepdims=True)
        std = epoch.std(axis=1, keepdims=True) + 1e-8
        epoch = (epoch - mean) / std

        clean_epochs.append(epoch)

    if len(clean_epochs) == 0:
        raise ValueError("No clean EEG epochs available after artifact rejection.")

    X = np.stack(clean_epochs)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    return X, clean_epochs, rejected_epochs, raw.info["sfreq"]


def predict(model, X):
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)

    stress_probs = probs[:, 1].cpu().numpy()
    stress_percent = float(np.mean(stress_probs) * 100)

    return stress_percent


st.set_page_config(
    page_title="EEG Stress Detection",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🧠 EEG Stress Detection</h1>
    <p style="text-align:center; font-size:18px;">
        Upload an EEG EDF file to detect stress level using EEGNet.
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload EEG EDF File", type=["edf"])

if uploaded_file is not None:
    try:
        ch19 = load_channels(CHANNELS_19_JSON)
        ch64 = load_channels(CHANNELS_64_JSON)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        raw_check = mne.io.read_raw_edf(temp_path, preload=False, verbose=False)

        model_path, required_channels, selected_model_name = select_model_and_channels(
            raw_check.ch_names,
            ch19,
            ch64
        )

        uploaded_file.seek(0)

        with st.spinner("Analyzing EEG signal..."):
            X, clean_epochs, rejected_epochs, sfreq = preprocess_edf(
                uploaded_file,
                required_channels
            )

            model = load_model(model_path)
            stress_percent = predict(model, X)
            level = stress_level(stress_percent)

        show_result_card(level)

        with st.expander("View technical details"):
            st.write("Selected model:", selected_model_name)
            st.write("Model file:", model_path)
            st.write("Channels used:", len(required_channels))
            st.write("Original sampling rate:", raw_check.info["sfreq"])
            st.write("Resampled frequency:", TARGET_SFREQ)
            st.write("Epoch size:", "2 sec = 256 samples")
            st.write("Clean epochs:", len(clean_epochs))
            st.write("Rejected epochs:", rejected_epochs)
            st.write("Internal stress percentage:", round(stress_percent, 2))

    except Exception as e:
        st.error("Prediction failed.")
        st.write(str(e))