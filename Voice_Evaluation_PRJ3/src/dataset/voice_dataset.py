import csv
import os
from torch.utils.data import Dataset

from audio_features import extract_features


class VoiceRankingDataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.audio_dir = audio_dir

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # CSV: videos/vid_1543.mp4
        video_path = row["video_path"]
        stem = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.audio_dir, f"{stem}.wav")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio: {audio_path}")

        features = extract_features(audio_path)
        rank = int(row["rank"])

        return {
            "features": features,   # [T, 4]
            "rank": rank
        }
    
if __name__ == "__main__":
    ds = VoiceRankingDataset(
        csv_path="voice_metadata.csv",
        audio_dir="data/audio"
    )

    print("Dataset length:", len(ds))

    sample = ds[0]
    print("Feature shape:", sample["features"].shape)
    print("Rank:", sample["rank"])

