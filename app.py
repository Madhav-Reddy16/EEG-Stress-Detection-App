import streamlit as st
import numpy as np
import mne
import torch
import torch.nn as nn
import tempfile
import os

st.set_page_config(page_title="EEG Stress Detection", page_icon="🧠", layout="wide")

st.title("🧠 EEG-Based Stress Detection App")
st.write("Upload an EEG EDF file to predict stress level using trained EEGNet models.")

MODEL_19_PATH = "eegnet_19ch.pt"
MODEL_64_PATH = "eegnet_64ch.pt"

CHANNELS_19 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "PZ", "P4", "T6", "O1", "O2"
]

CHANNELS_64 = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8", "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2", "IZ"
]


class EEGNet(nn.Module):
    def __init__(self, chans, samples, num_classes=2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            out = self.block2(self.block1(dummy))
            self.flatten_size = out.reshape(1, -1).shape[1]

        self.classifier = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def clean_channel_name(ch):
    ch = ch.upper().strip()

    # Remove common EEG prefixes
    ch = ch.replace("EEG ", "")
    ch = ch.replace("EEG", "")

    # Remove reference suffixes
    reference_suffixes = [
        "-LE", "-REF", "-AVG", "-A1", "-A2",
        "_LE", "_REF", "_AVG", "_A1", "_A2",
        " LE", " REF", " AVG", " A1", " A2"
    ]

    for suffix in reference_suffixes:
        ch = ch.replace(suffix, "")

    # Remove symbols
    ch = (
        ch.replace(".", "")
          .replace("-", "")
          .replace("_", "")
          .replace(" ", "")
    )

    # Final standard mapping
    standard_map = {
        "FP1": "FP1", "FP2": "FP2", "FPZ": "FPZ",
        "AF7": "AF7", "AF3": "AF3", "AFZ": "AFZ", "AF4": "AF4", "AF8": "AF8",
        "F7": "F7", "F5": "F5", "F3": "F3", "F1": "F1", "FZ": "FZ",
        "F2": "F2", "F4": "F4", "F6": "F6", "F8": "F8",
        "FT7": "FT7", "FC5": "FC5", "FC3": "FC3", "FC1": "FC1",
        "FCZ": "FCZ", "FC2": "FC2", "FC4": "FC4", "FC6": "FC6", "FT8": "FT8",
        "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6",
        "T7": "T7", "T8": "T8", "T9": "T9", "T10": "T10",
        "C5": "C5", "C3": "C3", "C1": "C1", "CZ": "CZ",
        "C2": "C2", "C4": "C4", "C6": "C6",
        "TP7": "TP7", "CP5": "CP5", "CP3": "CP3", "CP1": "CP1",
        "CPZ": "CPZ", "CP2": "CP2", "CP4": "CP4", "CP6": "CP6", "TP8": "TP8",
        "P7": "P7", "P5": "P5", "P3": "P3", "P1": "P1",
        "PZ": "PZ", "P2": "P2", "P4": "P4", "P6": "P6", "P8": "P8",
        "PO7": "PO7", "PO3": "PO3", "POZ": "POZ", "PO4": "PO4", "PO8": "PO8",
        "O1": "O1", "OZ": "OZ", "O2": "O2", "IZ": "IZ"
    }

    return standard_map.get(ch, ch)


def apply_19ch_alias(ch):
    alias_map = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6"
    }
    return alias_map.get(ch, ch)


def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )

        n_channels = checkpoint.get("n_channels", None)
        n_times = checkpoint.get("n_times", 256)
        sfreq = checkpoint.get("sfreq", 128)

    else:
        state_dict = checkpoint
        n_channels = None
        n_times = 256
        sfreq = 128

    if n_channels is None:
        first_weight = state_dict["block1.2.weight"]
        n_channels = first_weight.shape[2]

    model = EEGNet(
        chans=n_channels,
        samples=n_times,
        num_classes=2
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, n_channels, n_times, sfreq


def get_stress_level(stress_percent):
    if stress_percent <= 30:
        return "Normal", "🟢"
    elif stress_percent <= 60:
        return "Moderate Stress", "🟡"
    elif stress_percent <= 80:
        return "High Stress", "🟠"
    else:
        return "Severe Stress", "🔴"


def precautions_panel(level):
    if level == "Normal":
        st.success("""
### 🟢 Normal
- Maintain regular sleep.
- Stay hydrated.
- Continue light physical activity.
- Avoid unnecessary screen overload.
""")
    elif level == "Moderate Stress":
        st.warning("""
### 🟡 Moderate Stress
- Take a 5–10 minute breathing break.
- Reduce caffeine intake.
- Take short walking breaks.
- Avoid continuous screen exposure.
""")
    elif level == "High Stress":
        st.error("""
### 🟠 High Stress
- Stop heavy work for a short time.
- Try deep breathing or meditation.
- Drink water and relax your body.
- Avoid multitasking.
""")
    else:
        st.error("""
### 🔴 Severe Stress
- Take immediate rest.
- Sit in a calm place.
- Avoid mental overload.
- Consult a healthcare professional if stress continues.
""")


uploaded_file = st.file_uploader("Upload EEG EDF File", type=["edf"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

        raw.rename_channels(lambda ch: clean_channel_name(ch))
        raw.pick_types(eeg=True)
        raw.rename_channels(lambda ch: clean_channel_name(ch))

raw.pick_types(eeg=True)

reference_channels = [
    "A1", "A2",
    "A1A2", "A2A1",
    "M1", "M2",
    "REF", "AVG",
    "ECG", "EKG",
    "EOG", "HEOG", "VEOG",
    "EMG"
]

channels_to_remove = [
    ch for ch in raw.ch_names
    if ch in reference_channels
]

if channels_to_remove:
    raw.drop_channels(channels_to_remove)

        total_channels = len(raw.ch_names)
        st.info(f"Detected EEG channels: {total_channels}")

        if total_channels < 19:
            st.error("Incompatible EEG file. Minimum 19 EEG channels required.")
            st.stop()

        elif 19 <= total_channels < 60:
            selected_model_type = "19ch"
            required_channels = CHANNELS_19
            model_path = MODEL_19_PATH

        else:
            selected_model_type = "64ch"
            required_channels = CHANNELS_64
            model_path = MODEL_64_PATH

        missing_channels = [ch for ch in required_channels if ch not in raw.ch_names]

        if selected_model_type == "19ch" and missing_channels:
            raw.rename_channels(lambda ch: apply_19ch_alias(ch))
            missing_channels = [ch for ch in required_channels if ch not in raw.ch_names]

        if missing_channels:
            st.error("Prediction failed. Required trained channels are missing.")
            st.write(f"Selected model: {selected_model_type}")
            st.write(missing_channels)
            st.stop()

        raw.pick_channels(required_channels)

        model, model_channels, model_samples, model_sfreq = load_model(model_path)

        raw.resample(model_sfreq, verbose=False)
        raw.filter(1, 40, verbose=False)
        raw.notch_filter(50, verbose=False)

        data = raw.get_data()

        data = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-8
        )

        epochs = []
        total_samples = data.shape[1]

        for start in range(0, total_samples - model_samples + 1, model_samples):
            epoch = data[:, start:start + model_samples]
            epochs.append(epoch)

        if len(epochs) == 0:
            st.error("EEG file is too short. Minimum 2 seconds required.")
            st.stop()

        epochs = np.array(epochs)

        if epochs.shape[1] != model_channels:
            st.error("Channel mismatch between EEG file and trained model.")
            st.write(f"EEG channels: {epochs.shape[1]}")
            st.write(f"Model expected: {model_channels}")
            st.stop()

        X = torch.tensor(epochs, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            stress_probs = probs[:, 1].numpy()
            stress_percent = float(np.mean(stress_probs) * 100)

        level, icon = get_stress_level(stress_percent)

        st.success(f"Compatible EEG file detected. Using {selected_model_type} model.")
        st.write(f"Total 2-second epochs created: {len(epochs)}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Stress Assessment")
            st.metric("Stress Percentage", f"{stress_percent:.2f}%")
            st.progress(int(stress_percent))

        with col2:
            st.subheader(f"{icon} Final Result")
            st.markdown(f"### Stress Level: **{level}**")
            st.markdown(f"### Model Used: **{selected_model_type} EEGNet**")

        st.divider()
        precautions_panel(level)

    except Exception as e:
        st.error("Prediction failed due to processing error.")
        st.write(str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
