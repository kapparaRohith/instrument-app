import matplotlib
matplotlib.use('Agg')

import os
import gradio as gr
from predict import predict, predict_timeline, instrument_names
import matplotlib.pyplot as plt


def analyze_quick(audio_path):
    if audio_path is None:
        return "Please upload an audio file."

    predicted, probs = predict(audio_path)

    if predicted:
        summary = "🎵 Instruments detected: " + ", ".join(sorted(predicted))
    else:
        summary = "No instruments detected confidently."

    return summary


def analyze_timeline(audio_path):
    if audio_path is None:
        return None

    # Larger hop = fewer model runs = faster, more reliable on limited hosting
    timeline = predict_timeline(audio_path, window_sec=10, hop_sec=5)

    cmap = plt.get_cmap('tab20')
    colors = {name: cmap(i % 20) for i, name in enumerate(instrument_names)}

    fig, ax = plt.subplots(figsize=(12, 7))
    for row_idx, name in enumerate(instrument_names):
        for entry in timeline:
            if name in entry['predicted']:
                ax.barh(row_idx, 5, left=entry['window_start'],
                         height=0.8, color=colors[name], edgecolor='none')
    ax.set_yticks(range(len(instrument_names)))
    ax.set_yticklabels(instrument_names)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Instrument Activity Timeline')
    ax.set_xlim(0, timeline[-1]['window_end'])
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    return fig


with gr.Blocks(title="🎶 Instrument Recognition") as demo:
    gr.Markdown("# 🎶 Instrument Recognition")
    gr.Markdown("Upload a song to see which instruments are playing.")

    audio_input = gr.Audio(type="filepath", label="Upload a song")

    with gr.Row():
        quick_btn = gr.Button("Detect Instruments", variant="primary")
        timeline_btn = gr.Button("Show Timeline (slower)")

    summary_output = gr.Textbox(label="Detected Instruments")
    timeline_output = gr.Plot(label="Instrument Timeline")

    quick_btn.click(fn=analyze_quick, inputs=audio_input, outputs=summary_output)
    timeline_btn.click(fn=analyze_timeline, inputs=audio_input, outputs=timeline_output)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        max_threads=1
    )
