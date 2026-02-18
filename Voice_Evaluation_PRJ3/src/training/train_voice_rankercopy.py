import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.nn import HuberLoss
from tqdm import tqdm

from src.dataset.voice_cached_dataset import VoiceCachedDataset
from src.model.voice_modelcopy import VoiceRankingModel


def collate_fn(batch):
    feats = [b["features"] for b in batch]
    scores = torch.tensor(
        [b["score"] for b in batch],
        dtype=torch.float32
    )

    feats = pad_sequence(feats, batch_first=True)
    return feats, scores


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = VoiceCachedDataset(
        cache_dir="data/feature_cache"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = VoiceRankingModel(input_dim=8).to(device)

    loss_fn = HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=100
        )

        for step, (feats, scores) in enumerate(pbar):

            feats = feats.to(device)
            scores = scores.to(device)

            pred, _ = model(feats)

            loss = loss_fn(pred, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}"
            })

        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), "voice_model.pt")
    print("Model saved to voice_model.pt")


if __name__ == "__main__":
    train()
