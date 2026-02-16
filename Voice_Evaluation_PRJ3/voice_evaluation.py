import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sounddevice as sd
from scipy.io.wavfile import write

from audio_features import extract_features
from src.model.voice_model import VoiceRankingModel

MODEL_PATH = "voice_model.pt"
SAMPLE_RATE = 16000


def load_model():
    model = VoiceRankingModel(input_dim=8)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def convert_to_wav(input_path, output_wav):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(output_wav):
        raise RuntimeError("FFmpeg conversion failed.")

    return output_wav


def record_audio(duration_sec, output_wav):
    print(f"Recording for {duration_sec} seconds...")
    audio = sd.rec(int(duration_sec * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    write(output_wav, SAMPLE_RATE, audio)
    print("Recording complete.")
    return output_wav


def prepare_input(path, index):
    path = path.strip().strip('"')
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        return path

    output_wav = f"converted_{index}.wav"
    return convert_to_wav(path, output_wav)


def full_analysis(model, wav_path,
                  window_sec=5,
                  stride_sec=2):

    features = extract_features(wav_path)

    energy = features[:, 0].numpy()
    zcr = features[:, 1].numpy()
    centroid = features[:, 2].numpy()
    f0 = features[:, 4].numpy()
    jitter = features[:, 6].numpy()

    frames_per_sec = SAMPLE_RATE / 160
    window_frames = int(window_sec * frames_per_sec)
    stride_frames = int(stride_sec * frames_per_sec)

    stability_scores = []
    stability_times = []

    for start in range(0, len(features) - window_frames, stride_frames):
        end = start + window_frames
        window = features[start:end].unsqueeze(0)

        with torch.no_grad():
            score = model(window).item()

        stability_scores.append(score)
        stability_times.append(start / frames_per_sec)

    stability_scores = np.array(stability_scores)
    stability_times = np.array(stability_times)

    times = np.arange(len(energy)) / frames_per_sec

    return {
        "time_full": times,
        "energy": energy,
        "pitch": f0,
        "stability_time": stability_times,
        "stability_score": stability_scores
    }


def smooth_signal(x, win=50):
    if len(x) < win:
        return x
    return np.convolve(x, np.ones(win)/win, mode='same')


def plot_single_report(results):

    t = results["time_full"]

    energy = smooth_signal(results["energy"], 100)
    pitch = smooth_signal(results["pitch"], 100)
    stability = results["stability_score"]
    stability_time = results["stability_time"]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(stability_time, stability, linewidth=2)
    axs[0].set_title("Voice Stability Score")

    axs[1].plot(t, pitch, linewidth=1.5)
    axs[1].set_title("Pitch Over Time")

    axs[2].plot(t, energy, linewidth=1.5)
    axs[2].set_title("Loudness Over Time")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nSummary")
    print(f"Mean Stability: {np.mean(stability):.4f}")
    print(f"Mean Pitch: {np.mean(pitch):.2f}")
    print(f"Mean Energy: {np.mean(energy):.4f}")
    print()


def plot_comparison_report(r1, r2):

    plt.figure(figsize=(12, 6))
    plt.plot(r1["stability_time"], r1["stability_score"], label="Input 1")
    plt.plot(r2["stability_time"], r2["stability_score"], label="Input 2")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Stability Score")
    plt.title("Stability Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nComparison")
    print(f"Input 1 Mean Stability: {np.mean(r1['stability_score']):.4f}")
    print(f"Input 2 Mean Stability: {np.mean(r2['stability_score']):.4f}")
    print(f"Difference (2 - 1): {(np.mean(r2['stability_score']) - np.mean(r1['stability_score'])):.4f}\n")


if __name__ == "__main__":

    model = load_model()

    print("Select analysis type:")
    print("1 - Single analysis")
    print("2 - Comparative analysis")
    analysis_mode = input("Enter 1 or 2: ")

    print("\nSelect input mode:")
    print("1 - Use file")
    print("2 - Record live")
    input_mode = input("Enter 1 or 2: ")

    if analysis_mode == "1":

        if input_mode == "1":
            path = input("Enter full path: ")
            wav = prepare_input(path, 1)
        else:
            wav = record_audio(20, "recorded_single.wav")

        results = full_analysis(model, wav)
        plot_single_report(results)

    elif analysis_mode == "2":

        if input_mode == "1":
            path1 = input("Enter path for input 1: ")
            path2 = input("Enter path for input 2: ")
            wav1 = prepare_input(path1, 1)
            wav2 = prepare_input(path2, 2)
        else:
            wav1 = record_audio(20, "recorded_1.wav")
            wav2 = record_audio(20, "recorded_2.wav")

        r1 = full_analysis(model, wav1)
        r2 = full_analysis(model, wav2)

        plot_comparison_report(r1, r2)

    else:
        raise ValueError("Invalid option.")
