import torch
import numpy as np
from scipy.stats import spearmanr

from src.dataset.voice_dataset import VoiceRankingDataset
from src.model.voice_model import VoiceRankingModel


def validate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VoiceRankingDataset(
        csv_path="voice_metadata.csv",
        audio_dir="data/audio"
    )

    model = VoiceRankingModel().to(device)
    model.load_state_dict(torch.load("voice_model.pt", map_location=device))
    model.eval()

    scores = []
    ranks = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample["features"].to(device)
            rank = sample["rank"]

            score = model(features).item()

            scores.append(score)
            ranks.append(-rank)

    corr, _ = spearmanr(scores, ranks)
    print("Spearman correlation:", corr)


if __name__ == "__main__":
    validate()

