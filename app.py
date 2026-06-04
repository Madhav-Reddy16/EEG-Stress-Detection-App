import streamlit as st
import numpy as np
import mne
import torch
import torch.nn as nn
import tempfile
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="EEG Stress Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 EEG-Based Stress Detection App")
st.write("Upload an EEG EDF file to detect stress level using trained EEGNet models.")


# =====================================================
# CHANNEL LISTS
# =====================================================
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
    "T7", "T8",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2"
]

CHANNELS_64 = list(dict.fromkeys(CHANNELS_64))


# =====================================================
# MODEL PATHS
# =====================================================
MODEL_19_PATH = "eegnet_19ch.pt"
MODEL_64_PATH = "eegnet_64ch.pt"


# =====================================================
# EEGNET MODEL
# =====================================================
class EEGNet(nn.Module):
    def __init__(self, chans, samples, num_classes=2):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            out = self.separableConv(self.depthwiseConv(self.firstconv(dummy)))
            self.flatten_size = out.shape[1] * out.shape[2] * out.shape[3]

        self.classifier = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


# =====================================================
# FUNCTIONS
# =====================================================
def clean_channel_name(ch):
    return (
        ch.replace(".", "")
          .replace(" ", "")
          .replace("-", "")
          .replace("_", "")
          .upper()
    )


def apply_19ch_alias(ch):
    alias_map = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6"
    }
    return alias_map.get(ch, ch)


def load_model(model_path, chans, samples):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Read saved metadata if available
    saved_chans = checkpoint.get("n_channels", chans) if isinstance(checkpoint, dict) else chans
    saved_samples = checkpoint.get("n_times", samples) if isinstance(checkpoint, dict) else samples

    model = EEGNet(
        chans=saved_chans,
        samples=saved_samples,
        num_classes=2
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, saved_chans, saved_samples


def get_stress_level(stress_percent):
    if stress_percent <= 30:
        return "Normal", "🟢"
    elif stress_percent <= 60:
        return "Moderate Stress", "🟡"
    elif stress_percent <= 80:
        return "High Stress", "🟠"
    else:
        return "Severe Stress", "🔴"


def show_stress_visual(stress_percent):
    st.metric("Stress Percentage", f"{stress_percent:.2f}%")
    st.progress(int(stress_percent))


def precautions_panel(level):
    if level == "Normal":
        st.success("""
### 🟢 Normal Stress Precautions
- Maintain regular sleep.
- Continue light exercise.
- Stay hydrated.
- Avoid unnecessary screen overload.
""")

    elif level == "Moderate Stress":
        st.warning("""
### 🟡 Moderate Stress Precautions
- Take a 5–10 minute breathing break.
- Reduce caffeine intake.
- Take short walking breaks.
- Avoid continuous screen exposure.
""")

    elif level == "High Stress":
        st.error("""
### 🟠 High Stress Precautions
- Stop heavy work for a short time.
- Try deep breathing or meditation.
- Drink water and relax your body.
- Avoid multitasking.
""")

    else:
        st.error("""
### 🔴 Severe Stress Precautions
- Take immediate rest.
- Sit in a calm place.
- Avoid mental overload.
- Consult a healthcare professional if stress continues.
""")


# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload EEG EDF File", type=["edf"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        # =====================================================
        # READ EDF
        # =====================================================
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

        raw.rename_channels(lambda ch: clean_channel_name(ch))
        raw.pick_types(eeg=True)

        total_channels = len(raw.ch_names)

        st.info(f"Detected EEG channels: {total_channels}")

        # =====================================================
        # MODEL SELECTION
        # =====================================================
        if total_channels < 19:
            st.error("Incompatible EEG file. Minimum 19 EEG channels are required.")
            st.stop()

        elif 19 <= total_channels < 60:
            selected_model_type = "19ch"
            required_channels = CHANNELS_19
            model_path = MODEL_19_PATH

        else:
            selected_model_type = "64ch"
            required_channels = CHANNELS_64
            model_path = MODEL_64_PATH

        # =====================================================
        # CHANNEL CHECK
        # =====================================================
        missing_channels = [
            ch for ch in required_channels
            if ch not in raw.ch_names
        ]

        # Alias conversion only for 19ch mismatch
        if selected_model_type == "19ch" and missing_channels:
            raw.rename_channels(lambda ch: apply_19ch_alias(ch))

            missing_channels = [
                ch for ch in required_channels
                if ch not in raw.ch_names
            ]

        if missing_channels:
            st.error("Prediction failed.")
            st.write(f"Selected model: {selected_model_type}")
            st.write("Missing trained channels:")
            st.write(missing_channels)
            st.stop()

        raw.pick_channels(required_channels)

        st.success(f"Compatible EEG file detected. Using {selected_model_type} model.")

        # =====================================================
        # LOAD MODEL FIRST TO READ MODEL SHAPE
        # =====================================================
        model, model_chans, model_samples = load_model(
            model_path=model_path,
            chans=len(required_channels),
            samples=256
        )

        # =====================================================
        # PREPROCESSING
        # =====================================================
        TARGET_SFREQ = 128
        EPOCH_SECONDS = 2
        TARGET_SAMPLES = model_samples

        raw.resample(TARGET_SFREQ, verbose=False)

        raw.filter(1, 40, verbose=False)
        raw.notch_filter(50, verbose=False)

        data = raw.get_data()

        data = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-8
        )

        # =====================================================
        # EPOCHING
        # =====================================================
        epochs = []
        total_samples = data.shape[1]

        for start in range(0, total_samples - TARGET_SAMPLES + 1, TARGET_SAMPLES):
            epoch = data[:, start:start + TARGET_SAMPLES]
            epochs.append(epoch)

        if len(epochs) == 0:
            st.error("EEG file duration is too short for prediction.")
            st.stop()

        epochs = np.array(epochs)

        if epochs.shape[1] != model_chans:
            st.error("Channel mismatch between uploaded EEG and trained model.")
            st.write(f"EEG channels after selection: {epochs.shape[1]}")
            st.write(f"Model expected channels: {model_chans}")
            st.stop()

        if epochs.shape[2] != model_samples:
            st.error("Sample mismatch between uploaded EEG and trained model.")
            st.write(f"Epoch samples: {epochs.shape[2]}")
            st.write(f"Model expected samples: {model_samples}")
            st.stop()

        st.write(f"Total 2-second epochs created: {len(epochs)}")

        X = torch.tensor(epochs, dtype=torch.float32).unsqueeze(1)

        # =====================================================
        # PREDICTION
        # =====================================================
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            stress_probs = probabilities[:, 1].numpy()
            stress_percent = float(np.mean(stress_probs) * 100)

        level, icon = get_stress_level(stress_percent)

        # =====================================================
        # RESULT DISPLAY
        # =====================================================
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Stress Assessment")
            show_stress_visual(stress_percent)

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
