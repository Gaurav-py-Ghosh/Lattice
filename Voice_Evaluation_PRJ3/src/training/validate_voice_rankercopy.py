import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from src.dataset.voice_dataset2 import VoiceRankingDataset
from src.model.voice_modelcopy import VoiceRankingModel


def validate():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VoiceRankingDataset(
        csv_path="recruitview - Copy.csv",
        audio_dir="C:\\Users\\krish\\OneDrive\\Desktop\\Voice_Evaluation_PRJ3\\data\\audio"
    )

    model = VoiceRankingModel(input_dim=8).to(device)
    model.load_state_dict(torch.load("voice_model.pt", map_location=device))
    model.eval()

    scores = []
    targets = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Validating", ncols=100):

            sample = dataset[i]
            features = sample["features"].to(device)
            target = sample["score"]

            score, _ = model(features)
            score = score.item()

            scores.append(score)
            targets.append(target)

    corr, _ = spearmanr(scores, targets)
    print("\nSpearman correlation:", corr)


if __name__ == "__main__":
    validate()
