import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import HuberLoss
from tqdm import tqdm
import soundfile as sf

from src.dataset.voice_wav_dataset import VoiceWavDataset
from src.model.voice_wav2vec_model import VoiceWav2VecModel


def collate_fn(batch):
    audios = torch.stack([b["audio"] for b in batch])
    scores = torch.tensor([b["score"] for b in batch], dtype=torch.float32)
    return audios, scores


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = VoiceWavDataset(
        csv_path="recruitview - Copy.csv",
        audio_dir="C:\\Users\\krish\\OneDrive\\Desktop\\Voice_Evaluation_PRJ3\\data\\audio"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = VoiceWav2VecModel().to(device)

    loss_fn = HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

        for audios, scores in pbar:

            audios = audios.to(device)
            scores = scores.to(device)

            pred, _ = model(audios)

            loss = loss_fn(pred, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}"
            })

        print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), "voice_wav2vec_model.pt")
    print("Model saved.")


if __name__ == "__main__":
    train()
