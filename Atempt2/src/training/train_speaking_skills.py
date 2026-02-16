import torch
from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import sys
import os
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.dataset.recruitview_dataset import RecruitViewDataset
from src.model.video_model import VideoModel
from src.training.losses import PairwiseRankingLoss, HuberLoss
from src.utils.metrics import calculate_spearman_correlation

# --- Configuration ---
# Hardcoded for now, will move to config.yaml later
METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata.jsonl"
VIDEO_ROOT = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\videos"
TARGET_COLUMN = 'speaking_skills'

# Recommended Safe Baseline Configuration
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
FREEZE_VIDEO_ENCODER_EPOCHS = 4
NUM_FRAMES = 16
NUM_WORKERS = 4

def train_speaking_skills():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPUs.")

    # 1. Data Loading
    full_dataset = RecruitViewDataset(
        METADATA_PATH,
        VIDEO_ROOT,
        TARGET_COLUMN,
        num_frames=NUM_FRAMES
    )
    
    # Split into train and validation (e.g., 80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Batch size: {BATCH_SIZE}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Model
    model = VideoModel().to(device)

    # 3. Loss Functions
    ranking_loss_fn = PairwiseRankingLoss()
    huber_loss_fn = HuberLoss()

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # 5. Training Loop
    best_val_spearman_corr = -1.0 # Initialize with a value lower than any possible correlation

    for epoch in range(NUM_EPOCHS):
        # Freeze/Unfreeze Video Encoder
        if epoch < FREEZE_VIDEO_ENCODER_EPOCHS:
            print(f"Freezing video encoder for epoch {epoch+1}")
            for param in model.video_encoder.parameters():
                param.requires_grad = False
        else:
            if epoch == FREEZE_VIDEO_ENCODER_EPOCHS: # Only print once
                print(f"Unfreezing video encoder from epoch {epoch+1}")
            for param in model.video_encoder.parameters():
                param.requires_grad = True

        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
        for batch_idx, (video_frames, targets) in enumerate(train_loop):
            video_frames = video_frames.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                predictions = model(video_frames)
                # Calculate combined loss
                ranking_loss = ranking_loss_fn(predictions, targets)
                huber_loss = huber_loss_fn(predictions, targets.unsqueeze(1)) # Huber expects (N, 1) or (N)
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
            for video_frames, targets in val_loop:
                video_frames = video_frames.to(device)
                targets = targets.to(device)

                predictions = model(video_frames)
                
                ranking_loss = ranking_loss_fn(predictions, targets)
                huber_loss = huber_loss_fn(predictions, targets.unsqueeze(1))
                total_loss = (0.9 * ranking_loss) + (0.1 * huber_loss)
                
                total_val_loss += total_loss.item()
                
                # Ensure we handle the last batch case where batch size might be 1
                all_predictions.extend(np.atleast_1d(predictions.squeeze().cpu().numpy()))
                all_targets.extend(np.atleast_1d(targets.squeeze().cpu().numpy()))
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate Spearman correlation on validation set
        val_spearman_corr = calculate_spearman_correlation(torch.tensor(all_predictions), torch.tensor(all_targets))

        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f} | Validation Spearman: {val_spearman_corr:.4f}")

        # 7. Checkpointing
        os.makedirs("checkpoints", exist_ok=True)
        # Save the best model based on validation Spearman correlation
        if val_spearman_corr > best_val_spearman_corr:
            best_val_spearman_corr = val_spearman_corr
            best_model_path = os.path.join("checkpoints", "best_speaking_skills_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with Spearman correlation: {val_spearman_corr:.4f}")

    print("Training complete!")
    print(f"Best validation Spearman correlation: {best_val_spearman_corr:.4f}")

if __name__ == '__main__':
    train_speaking_skills()
