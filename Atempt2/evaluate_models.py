import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
import os
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.dataset.recruitview_dataset import RecruitViewDataset
from src.model.video_model import VideoModel
from src.utils.metrics import calculate_spearman_correlation

# --- Configuration ---
METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata.jsonl"
VIDEO_ROOT = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\videos"
NUM_FRAMES = 16
BATCH_SIZE = 8
NUM_WORKERS = 4

def evaluate_model(checkpoint_path, target_column, model_name):
    """
    Evaluate a saved model checkpoint on the validation set.
    
    Args:
        checkpoint_path (str): Path to the saved .pth checkpoint
        target_column (str): Target column to evaluate on
        model_name (str): Display name for the model
    
    Returns:
        float: Spearman correlation on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Target: {target_column}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return None
    
    # 1. Load Dataset
    print("\n📊 Loading dataset...")
    full_dataset = RecruitViewDataset(
        METADATA_PATH,
        VIDEO_ROOT,
        target_column,
        num_frames=NUM_FRAMES
    )
    
    # Split into train and validation (same split as training)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use same random seed as training to get the same split
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✓ Validation set size: {len(val_dataset)} samples")
    
    # 2. Load Model Architecture
    print("\n🏗️  Loading model architecture...")
    model = VideoModel(hidden_dim=256, num_heads=4)
    
    # 3. Load Checkpoint
    print(f"📥 Loading checkpoint...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    
    # 4. Evaluation Loop
    print("\n🔍 Running evaluation...")
    all_predictions = []
    all_targets = []
    
    val_loop = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for video_frames, targets in val_loop:
            video_frames = video_frames.to(device)
            targets = targets.to(device)
            
            predictions = model(video_frames)
            
            # Collect predictions and targets
            all_predictions.extend(np.atleast_1d(predictions.squeeze().cpu().numpy()))
            all_targets.extend(np.atleast_1d(targets.squeeze().cpu().numpy()))
    
    # 5. Calculate Spearman Correlation
    spearman_corr = calculate_spearman_correlation(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets)
    )
    
    print(f"\n{'='*60}")
    print(f"✅ RESULTS for {model_name}")
    print(f"{'='*60}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print(f"{'='*60}\n")
    
    return spearman_corr


def main():
    """
    Evaluate all saved model checkpoints.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION SCRIPT")
    print("="*60)
    
    results = {}
    
    # Model 1: Facial Expression Model
    results['Facial Expression'] = evaluate_model(
        checkpoint_path="checkpoints/best_facial_expression_model.pth",
        target_column="facial_expression",
        model_name="Facial Expression Model"
    )
    
    # Model 2: Confidence Score Model (original)
    results['Confidence (best_model - Copy)'] = evaluate_model(
        checkpoint_path="checkpoints/best_model - Copy.pth",
        target_column="confidence_score",
        model_name="Confidence Score Model (best_model - Copy.pth)"
    )
    
    # Model 3: Confidence Score Model (epoch 6)
    results['Confidence (epoch 6)'] = evaluate_model(
        checkpoint_path="checkpoints/videoMAE_confidence_ranker_epoch6.pth",
        target_column="confidence_score",
        model_name="Confidence Score Model (videoMAE_confidence_ranker_epoch6.pth)"
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY OF ALL MODELS")
    print("="*60)
    for model_name, spearman in results.items():
        if spearman is not None:
            print(f"{model_name:.<50} {spearman:.4f}")
        else:
            print(f"{model_name:.<50} ERROR")
    print("="*60 + "\n")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1] if x[1] is not None else -1)
    if best_model[1] is not None:
        print(f"🏆 Best performing model: {best_model[0]} (Spearman: {best_model[1]:.4f})")
    

if __name__ == '__main__':
    main()
