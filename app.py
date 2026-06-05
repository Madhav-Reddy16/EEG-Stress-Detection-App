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
    def __init__(self, chans, samples, num_classes=2, F1=16, D=2, F2=32, kern_length=64):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
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
    ch = ch.replace("EEG ", "").replace("EEG", "")

    reference_suffixes = [
        "-LE", "-REF", "-AVG", "-A1", "-A2",
        "_LE", "_REF", "_AVG", "_A1", "_A2",
        " LE", " REF", " AVG", " A1", " A2"
    ]

    for suffix in reference_suffixes:
        ch = ch.replace(suffix, "")

    ch = (
        ch.replace(".", "")
          .replace("-", "")
          .replace("_", "")
          .replace(" ", "")
    )

    return ch


def apply_19ch_alias(ch):
    alias_map = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6"
    }
    return alias_map.get(ch, ch)
def adapt_19ch_flexible(raw, required_channels):
    """
    19ch flexible adaptation:
    1. If all trained channels are available, process normally.
    2. If some trained channels are missing, use available channels and fill missing channels with zeros.
    """

    raw.rename_channels(lambda ch: clean_channel_name(ch))
    raw.rename_channels(lambda ch: apply_19ch_alias(ch))

    available_channels = [
        ch for ch in required_channels
        if ch in raw.ch_names
    ]

    missing_channels = [
        ch for ch in required_channels
        if ch not in raw.ch_names
    ]

    # Case 1: all required channels are available
    if len(missing_channels) == 0:
        raw.pick_channels(required_channels)
        return raw, missing_channels, "normal"

    # Case 2: some channels are missing
    if len(available_channels) == 0:
        return raw, missing_channels, "failed"

    data = raw.get_data(picks=available_channels)
    sfreq = raw.info["sfreq"]

    full_data = np.zeros((len(required_channels), data.shape[1]))

    for i, ch in enumerate(required_channels):
        if ch in available_channels:
            src_idx = available_channels.index(ch)
            full_data[i] = data[src_idx]

    info = mne.create_info(
        ch_names=required_channels,
        sfreq=sfreq,
        ch_types="eeg"
    )

    new_raw = mne.io.RawArray(full_data, info, verbose=False)

    return new_raw, missing_channels, "partial"

def adapt_64ch_channels(raw, required_channels):
    raw.rename_channels(lambda ch: clean_channel_name(ch))

    reference_channels = [
        "A1", "A2", "A1A2", "A2A1",
        "M1", "M2",
        "REF", "LE", "AVG",
        "ECG", "EKG",
        "EOG", "HEOG", "VEOG",
        "EMG", "STI", "STIM", "STATUS"
    ]

    channels_to_remove = [ch for ch in raw.ch_names if ch in reference_channels]

    if channels_to_remove:
        raw.drop_channels(channels_to_remove)

    raw.rename_channels(lambda ch: clean_channel_name(ch))

    missing_channels = [ch for ch in required_channels if ch not in raw.ch_names]

    if not missing_channels:
        raw.pick_channels(required_channels)
        return raw, []

    alias_64 = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8"
    }

    rename_dict = {}

    for old_ch, new_ch in alias_64.items():
        if old_ch in raw.ch_names and new_ch in required_channels and new_ch not in raw.ch_names:
            rename_dict[old_ch] = new_ch

    if rename_dict:
        raw.rename_channels(rename_dict)

    missing_channels = [ch for ch in required_channels if ch not in raw.ch_names]

    if missing_channels:
        return raw, missing_channels

    raw.pick_channels(required_channels)
    return raw, []

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
        sfreq = checkpoint.get("sfreq", 128)
    else:
        state_dict = checkpoint
        sfreq = 128

    F1 = state_dict["block1.0.weight"].shape[0]
    kern_length = state_dict["block1.0.weight"].shape[3]
    depth_filters = state_dict["block1.2.weight"].shape[0]
    n_channels = state_dict["block1.2.weight"].shape[2]
    F2 = state_dict["block2.1.weight"].shape[0]
    D = depth_filters // F1

    expected_features = state_dict["classifier.weight"].shape[1]

    possible_samples = [128, 256, 512, 1024]

    correct_samples = None

    for samples in possible_samples:
        temp_model = EEGNet(
            chans=n_channels,
            samples=samples,
            num_classes=2,
            F1=F1,
            D=D,
            F2=F2,
            kern_length=kern_length
        )

        if temp_model.classifier.in_features == expected_features:
            correct_samples = samples
            break

    if correct_samples is None:
        st.error("Could not detect model input sample size.")
        st.stop()

    model = EEGNet(
        chans=n_channels,
        samples=correct_samples,
        num_classes=2,
        F1=F1,
        D=D,
        F2=F2,
        kern_length=kern_length
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, n_channels, correct_samples, sfreq

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

        reference_channels = [
            "A1", "A2", "A1A2", "A2A1",
            "M1", "M2",
            "REF", "LE", "AVG",
            "ECG", "EKG",
            "EOG", "HEOG", "VEOG",
            "EMG", "STI", "STIM", "STATUS"
        ]

        channels_to_remove = [ch for ch in raw.ch_names if ch in reference_channels]

        if channels_to_remove:
            raw.drop_channels(channels_to_remove)

        total_channels = len(raw.ch_names)
        st.info(f"Detected EEG channels after cleaning: {total_channels}")

        if total_channels < 19:
            st.error("Incompatible EEG file. Minimum 19 EEG channels required.")
            st.stop()

        elif 19 <= total_channels < 60:

    selected_model_type = "19ch"
    required_channels = CHANNELS_19
    model_path = MODEL_19_PATH

    raw, missing_channels, channel_mode = adapt_19ch_flexible(
        raw,
        required_channels
    )

    if channel_mode == "normal":
        st.success("All trained 19 channels are available. Processing normally.")

    elif channel_mode == "partial":
        st.warning(
            f"{len(missing_channels)} trained channels are missing. "
            "Prediction is based on available EEG channels only."
        )
        st.write("Missing channels filled with zeros:")
        st.write(missing_channels)

    else:
        st.error("Prediction failed. No compatible trained channels were found.")
        st.stop()

    else:
            selected_model_type = "64ch"
            required_channels = CHANNELS_64
            model_path = MODEL_64_PATH

            raw, missing_channels = adapt_64ch_channels(raw, required_channels)

   if missing_channels:
                st.error("Prediction failed. Required trained channels are missing.")
                st.write(f"Selected model: {selected_model_type}")
                st.write("Available channels after cleaning:")
                st.write(raw.ch_names)
                st.write("Missing channels:")
                st.write(missing_channels)
                st.stop()

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
