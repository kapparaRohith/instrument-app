import json
import numpy as np
import librosa
import torch
from model import build_model

# ---- Load everything once, at startup ----
with open('models/norm_stats.json') as f:
    norm_stats = json.load(f)

with open('models/resnet_optimal_thresholds.json') as f:
    thresholds = json.load(f)

with open('models/class-map.json') as f:
    class_map = json.load(f)

instrument_names = list(class_map.keys())

device = torch.device('cpu')
model = build_model(num_classes=len(instrument_names))
model.load_state_dict(torch.load('models/best_resnet.pth', map_location=device))
model.eval()

TARGET_FRAMES = 862

def fix_length(spec, target_frames=TARGET_FRAMES):
    current = spec.shape[1]
    if current == target_frames:
        return spec
    elif current < target_frames:
        pad_width = target_frames - current
        return np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=spec.min())
    else:
        return spec[:, :target_frames]

def predict(audio_path):
    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # 2. Mel-spectrogram (same settings as training)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Fix length to match training shape
    mel_spec_db = fix_length(mel_spec_db)

    # 4. Normalize using training statistics
    mel_norm = (mel_spec_db - norm_stats['mean']) / norm_stats['std']

    # 5. Convert to tensor: (1 channel, 128, 862) -> add batch dim -> (1, 1, 128, 862)
    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 6. Run through model
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    # 7. Apply per-instrument thresholds
    results = {}
    predicted = []
    for i, name in enumerate(instrument_names):
        prob = float(probs[i])
        results[name] = prob
        if prob >= thresholds[name]:
            predicted.append(name)

    return predicted, results

def predict_timeline(audio_path, window_sec=10, hop_sec=1):
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = len(y) / sr
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    timeline = []
    start_sample = 0

    while start_sample < len(y):
        end_sample = start_sample + window_samples
        segment = y[start_sample:end_sample]

        # Pad the last (short) window with silence so it still matches the model's expected length
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = fix_length(mel_spec_db)
        mel_norm = (mel_spec_db - norm_stats['mean']) / norm_stats['std']
        tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        predicted = [name for i, name in enumerate(instrument_names) if probs[i] >= thresholds[name]]

        start_time = start_sample / sr
        timeline.append({
            'window_start': round(start_time, 1),
            'window_end': round(start_time + window_sec, 1),
            'predicted': predicted,
            'probs': {name: float(probs[i]) for i, name in enumerate(instrument_names)}
        })

        start_sample += hop_samples

    return timeline


def print_timeline(audio_path):
    timeline = predict_timeline(audio_path, window_sec=10, hop_sec=1)
    print(f"{'Time':>12s} | Predicted instruments")
    print("-" * 60)
    for entry in timeline:
        t = f"{entry['window_start']:.0f}s"
        instruments = ", ".join(entry['predicted']) if entry['predicted'] else "(none confident)"
        print(f"{t:>12s} | {instruments}")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_timeline(audio_path, window_sec=10, hop_sec=1, save_path='timeline.png'):
    timeline = predict_timeline(audio_path, window_sec=window_sec, hop_sec=hop_sec)

    # Build a simple color palette, one color per instrument
    cmap = plt.get_cmap('tab20')
    colors = {name: cmap(i % 20) for i, name in enumerate(instrument_names)}

    fig, ax = plt.subplots(figsize=(14, 8))

    for row_idx, name in enumerate(instrument_names):
        for entry in timeline:
            if name in entry['predicted']:
                ax.barh(row_idx, hop_sec, left=entry['window_start'],
                        height=0.8, color=colors[name], edgecolor='none')

    ax.set_yticks(range(len(instrument_names)))
    ax.set_yticklabels(instrument_names)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Instrument Activity Timeline')
    ax.set_xlim(0, timeline[-1]['window_end'])
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    import sys
    plot_timeline(sys.argv[1])
