import gradio as gr
from predict import predict, predict_timeline, instrument_names
import matplotlib.pyplot as plt

def analyze(audio_path):
    if audio_path is None:
        return "Please upload an audio file.", None

    # Overall prediction (whole clip)
    predicted, probs = predict(audio_path)

    if predicted:
        summary = "🎵 Instruments detected: " + ", ".join(sorted(predicted))
    else:
        summary = "No instruments detected confidently."

    # Timeline
    timeline = predict_timeline(audio_path, window_sec=10, hop_sec=1)

    cmap = plt.get_cmap('tab20')
    colors = {name: cmap(i % 20) for i, name in enumerate(instrument_names)}

    fig, ax = plt.subplots(figsize=(12, 7))
    for row_idx, name in enumerate(instrument_names):
        for entry in timeline:
            if name in entry['predicted']:
                ax.barh(row_idx, 1, left=entry['window_start'],
                         height=0.8, color=colors[name], edgecolor='none')
    ax.set_yticks(range(len(instrument_names)))
    ax.set_yticklabels(instrument_names)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Instrument Activity Timeline')
    ax.set_xlim(0, timeline[-1]['window_end'])
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    return summary, fig


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(type="filepath", label="Upload a song"),
    outputs=[
        gr.Textbox(label="Detected Instruments"),
        gr.Plot(label="Instrument Timeline")
    ],
    title="🎶 Instrument Recognition",
    description="Upload a song to see which instruments are playing, and when, throughout the track.",
)

import os

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
