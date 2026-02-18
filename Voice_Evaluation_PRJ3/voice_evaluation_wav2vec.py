import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from scipy.stats import percentileofscore

from src.model.voice_wav2vec_model import VoiceWav2VecModel


MODEL_PATH = "voice_wav2vec_model.pt"
CSV_PATH = "recruitview - Copy.csv"
SAMPLE_RATE = 16000
MAX_SECONDS = 15


# -------------------------
# Utility
# -------------------------

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VoiceWav2VecModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def load_dataset_scores():
    import csv
    scores = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["speaking_skills"]))
    return np.array(scores)


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


# -------------------------
# Core Analysis
# -------------------------

def load_waveform(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    max_len = SAMPLE_RATE * MAX_SECONDS

    if len(wav) > max_len:
        wav = wav[:max_len]
    else:
        pad = max_len - len(wav)
        wav = np.pad(wav, (0, pad))

    return torch.tensor(wav)


def predict_score(model, device, waveform):
    waveform = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        score, _ = model(waveform)
    return score.item()


def sliding_window_analysis(model, device, waveform,
                            window_sec=5,
                            stride_sec=2):

    window_len = SAMPLE_RATE * window_sec
    stride_len = SAMPLE_RATE * stride_sec

    scores = []
    times = []

    for start in range(0, len(waveform) - window_len, stride_len):
        segment = waveform[start:start + window_len]
        segment = segment.unsqueeze(0).to(device)

        with torch.no_grad():
            score, _ = model(segment)

        scores.append(score.item())
        times.append(start / SAMPLE_RATE)

    return np.array(times), np.array(scores)


def extract_pitch_energy(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    energy = np.sqrt(np.convolve(wav**2,
                                 np.ones(400)/400,
                                 mode='same'))

    # simple pitch proxy via zero-crossing
    zcr = np.abs(np.diff(np.sign(wav))).astype(float)

    return energy, zcr


# -------------------------
# Plotting
# -------------------------

def plot_single_report(score, percentile,
                       times, window_scores,
                       energy, pitch_proxy):

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(times, window_scores, linewidth=2)
    axs[0].set_title("Speaking Skills (Sliding Window)")

    axs[1].plot(energy, linewidth=1.2)
    axs[1].set_title("Energy (Loudness Proxy)")

    axs[2].plot(pitch_proxy, linewidth=1.2)
    axs[2].set_title("Pitch Proxy (ZCR-based)")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nOverall Speaking Skills Score:", round(score, 4))
    print("Percentile vs Dataset:", round(percentile, 2), "%")
    print()


def plot_comparison_report(r1, r2):

    plt.figure(figsize=(12, 6))
    plt.plot(r1["times"], r1["window_scores"], label="Input 1")
    plt.plot(r2["times"], r2["window_scores"], label="Input 2")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Speaking Skills")
    plt.title("Speaking Skills Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nComparison:")
    print("Input 1 Score:", round(r1["score"], 4))
    print("Input 2 Score:", round(r2["score"], 4))
    print("Difference (2 - 1):", round(r2["score"] - r1["score"], 4))
    print()


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    model, device = load_model()
    dataset_scores = load_dataset_scores()

    print("Select analysis type:")
    print("1 - Single analysis")
    print("2 - Comparative analysis")
    analysis_mode = input("Enter 1 or 2: ")

    print("\nSelect input mode:")
    print("1 - Use file")
    print("2 - Record live")
    input_mode = input("Enter 1 or 2: ")

    def process_input(path, idx):
        wav_path = prepare_input(path, idx)
        waveform = load_waveform(wav_path)
        score = predict_score(model, device, waveform)
        percentile = percentileofscore(dataset_scores, score)

        times, window_scores = sliding_window_analysis(
            model, device, waveform
        )

        energy, pitch_proxy = extract_pitch_energy(wav_path)

        return {
            "score": score,
            "percentile": percentile,
            "times": times,
            "window_scores": window_scores,
            "energy": energy,
            "pitch": pitch_proxy
        }

    if analysis_mode == "1":

        if input_mode == "1":
            path = input("Enter full path: ")
        else:
            path = record_audio(15, "recorded_single.wav")

        result = process_input(path, 1)

        plot_single_report(
            result["score"],
            result["percentile"],
            result["times"],
            result["window_scores"],
            result["energy"],
            result["pitch"]
        )

    elif analysis_mode == "2":

        if input_mode == "1":
            path1 = input("Enter path for input 1: ")
            path2 = input("Enter path for input 2: ")
        else:
            path1 = record_audio(15, "recorded_1.wav")
            path2 = record_audio(15, "recorded_2.wav")

        r1 = process_input(path1, 1)
        r2 = process_input(path2, 2)

        plot_comparison_report(r1, r2)

    else:
        raise ValueError("Invalid option.")
