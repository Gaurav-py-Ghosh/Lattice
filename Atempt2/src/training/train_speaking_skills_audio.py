import torch
from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
import os
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.dataset.audio_dataset import RecruitViewAudioDataset
from src.model.audio_model import AudioModel
from src.training.losses import PairwiseRankingLoss, HuberLoss
from src.utils.metrics import calculate_spearman_correlation

# --- Configuration ---
METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata_with_audio.jsonl"
AUDIO_ROOT = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\audio"
TARGET_COLUMN = 'speaking_skills'

# Fine-tuning hyperparameters for the audio model
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
FREEZE_AUDIO_ENCODER_EPOCHS = 4
NUM_WORKERS = 4 # Can be set to 0 on Windows if multiprocessing issues arise

def collate_fn(batch):
    """
    Custom collate function to pad audio waveforms to the same length in a batch.
    """
    waveforms, targets = zip(*batch)
    
    # Squeeze the channel dimension before padding
    waveforms = [w.squeeze(0) for w in waveforms]
    
    # Pad sequences to the length of the longest sequence in the batch
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    
    return padded_waveforms, targets

def train_speaking_skills_audio():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPUs.")

    # 1. Data Loading
    full_dataset = RecruitViewAudioDataset(
        METADATA_PATH,
        AUDIO_ROOT,
        TARGET_COLUMN
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Batch size: {BATCH_SIZE}")

    # Use the custom collate_fn in the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    # 2. Model
    model = AudioModel().to(device)

    # 3. Loss Functions
    ranking_loss_fn = PairwiseRankingLoss()
    huber_loss_fn = HuberLoss()

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # 5. Training Loop
    best_val_spearman_corr = -1.0

    for epoch in range(NUM_EPOCHS):
        if epoch < FREEZE_AUDIO_ENCODER_EPOCHS:
            print(f"Freezing audio encoder for epoch {epoch+1}")
            for param in model.audio_encoder.parameters():
                param.requires_grad = False
        else:
            if epoch == FREEZE_AUDIO_ENCODER_EPOCHS:
                print(f"Unfreezing audio encoder from epoch {epoch+1}")
            for param in model.audio_encoder.parameters():
                param.requires_grad = True

        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
        for audio_waveforms, targets in train_loop:
            audio_waveforms = audio_waveforms.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                predictions = model(audio_waveforms)
                ranking_loss = ranking_loss_fn(predictions, targets)
                huber_loss = huber_loss_fn(predictions, targets.unsqueeze(1))
                total_loss = (0.9 * ranking_loss) + (0.1 * huber_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += total_loss.item()
            train_loop.set_postfix(loss=total_loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f}")

        # 6. Validation Loop
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Validation")
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for audio_waveforms, targets in val_loop:
                audio_waveforms = audio_waveforms.to(device)
                targets = targets.to(device)

                predictions = model(audio_waveforms)
                
                ranking_loss = ranking_loss_fn(predictions, targets)
                huber_loss = huber_loss_fn(predictions, targets.unsqueeze(1))
                total_loss = (0.9 * ranking_loss) + (0.1 * huber_loss)
                
                total_val_loss += total_loss.item()
                
                all_predictions.extend(np.atleast_1d(predictions.squeeze().cpu().numpy()))
                all_targets.extend(np.atleast_1d(targets.squeeze().cpu().numpy()))
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_spearman_corr = calculate_spearman_correlation(torch.tensor(all_predictions), torch.tensor(all_targets))

        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f} | Validation Spearman: {val_spearman_corr:.4f}")

        # 7. Checkpointing
        os.makedirs("checkpoints", exist_ok=True)
        if val_spearman_corr > best_val_spearman_corr:
            best_val_spearman_corr = val_spearman_corr
            best_model_path = os.path.join("checkpoints", "best_speaking_skills_audio_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with Spearman correlation: {val_spearman_corr:.4f}")

    print("Training complete!")
    print(f"Best validation Spearman correlation: {best_val_spearman_corr:.4f}")

if __name__ == '__main__':
    train_speaking_skills_audio()
