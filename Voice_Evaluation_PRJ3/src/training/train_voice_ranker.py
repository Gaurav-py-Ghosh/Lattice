import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from tqdm import tqdm

from src.dataset.pair_dataset import VoicePairDataset
from src.model.voice_model import VoiceRankingModel
from src.training.losses import RankingLoss


def collate_fn(batch):
    feats_a = [b["features_a"] for b in batch]
    feats_b = [b["features_b"] for b in batch]

    feats_a = pad_sequence(feats_a, batch_first=True)
    feats_b = pad_sequence(feats_b, batch_first=True)

    return feats_a, feats_b


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VoicePairDataset(
        csv_path="voice_metadata.csv",
        audio_dir="data/audio",
        num_pairs=50000
    )
    print("Dataset initialized with", len(dataset), "pairs")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = VoiceRankingModel(input_dim=8).to(device)
    loss_fn = RankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    running_loss = 0.0
    max_steps = 300

    pbar = tqdm(loader, total=max_steps, desc="Training", ncols=100)

    for step, (fa, fb) in enumerate(pbar):
        if step >= max_steps:
            break

        fa = fa.to(device)
        fb = fb.to(device)

        sa = model(fa)
        sb = model(fb)

        loss = loss_fn(sa, sb)

        optimizer.zero_grad()
        loss.backward()

        gnorm = grad_norm(model)
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (step + 1)

        lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "gnorm": f"{gnorm:.2f}",
            "lr": lr
        })

    torch.save(model.state_dict(), "voice_model.pt")
    print("\nModel saved to voice_model.pt")


if __name__ == "__main__":
    train()
