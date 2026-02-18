import os
import csv
import torch
import soundfile as sf
from torch.utils.data import Dataset


class VoiceWavDataset(Dataset):

    def __init__(self, csv_path, audio_dir, max_seconds=15):
        self.audio_dir = audio_dir
        self.max_length = 16000 * max_seconds  # 16kHz

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        row = self.rows[idx]

        # Strip any folder structure from CSV
        filename = os.path.basename(row["file_name"])
        stem = os.path.splitext(filename)[0]

        audio_path = os.path.join(self.audio_dir, f"{stem}.wav")

        wav, sr = sf.read(audio_path, dtype="float32")

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        waveform = torch.from_numpy(wav)

        # Trim or pad
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            pad = self.max_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        score = float(row["speaking_skills"])

        return {
            "audio": waveform,
            "score": score
        }
