import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from src.dataset.voice_wav_dataset import VoiceWavDataset
from src.model.voice_wav2vec_model import VoiceWav2VecModel


def validate():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VoiceWavDataset(
        csv_path="recruitview - Copy.csv",
        audio_dir="C:\\Users\\krish\\OneDrive\\Desktop\\Voice_Evaluation_PRJ3\\data\\audio"
    )

    model = VoiceWav2VecModel().to(device)
    model.load_state_dict(torch.load("voice_wav2vec_model.pt", map_location=device))
    model.eval()

    scores = []
    targets = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Validating", ncols=100):

            audio = sample["audio"].unsqueeze(0).to(device)
            target = sample["score"]

            pred, _ = model(audio)
            pred = pred.item()

            scores.append(pred)
            targets.append(target)

    corr, _ = spearmanr(scores, targets)
    print("\nSpearman correlation:", corr)


if __name__ == "__main__":
    validate()
