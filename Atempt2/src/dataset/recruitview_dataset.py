import os
import json
import decord
from decord import VideoReader, cpu
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
import numpy as np

# Set decord's bridge to PyTorch for direct tensor output
decord.bridge.set_bridge('torch')

class RecruitViewDataset(Dataset):
    """
    PyTorch Dataset for the RecruitView dataset.
    Reads video frames, uses the VideoMAE processor, and returns a single target score.
    """
    def __init__(self, metadata_path, video_root, target_column='confidence_score', num_frames=16):
        """
        Args:
            metadata_path (str): Path to the metadata.jsonl file.
            video_root (str): Path to the directory containing the video files.
            target_column (str): The name of the target column to use.
            num_frames (int): The number of frames to resample each video to.
        """
        self.video_root = video_root
        self.target_column = target_column
        self.num_frames = num_frames
        
        # Get the correct processor for the model directly from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        # Load and clean metadata
        data = []
        with open(metadata_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.metadata = pd.DataFrame(data)
        
        # Handle NaN values in the target column
        self.metadata[self.target_column] = self.metadata[self.target_column].fillna(0.0)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (video_tensor, target_score)
                   video_tensor is of shape (C, T, H, W)
        """
        video_info = self.metadata.iloc[idx]
        
        relative_video_path = video_info['file_name']
        if relative_video_path.startswith('videos/'):
            relative_video_path = relative_video_path[len('videos/'):]
        video_path = os.path.join(self.video_root, relative_video_path)

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Resample frames: take N frames evenly spaced
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            video_frames = vr.get_batch(indices) # This is a torch tensor (T, H, W, C)
            
        except Exception as e:
            print(f"Error reading video file {video_path}: {e}")
            # Return a dummy tensor if a video is corrupted
            # Processor expects a list of frames in H, W, C format
            video_frames = torch.zeros((self.num_frames, 224, 224, 3), dtype=torch.uint8)

        # The processor expects a list of numpy arrays (H, W, C)
        video_frames_list = [frame.numpy() for frame in video_frames]
        
        # Process the video frames. This handles everything: resizing, normalization, and permutation.
        # The output format will be (C, T, H, W)
        processed_video = self.processor(video_frames_list, return_tensors="pt")
        
        # Get the tensor from the processor's output
        video_tensor = processed_video['pixel_values'].squeeze(0)

        # Get target score
        target_score = torch.tensor(video_info[self.target_column], dtype=torch.float32)
        
        return video_tensor, target_score

if __name__ == '__main__':
    # Example usage and sanity check
    METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata.jsonl"
    VIDEO_ROOT = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\videos"

    # Create dataset
    dataset = RecruitViewDataset(METADATA_PATH, VIDEO_ROOT, target_column='confidence_score')
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    video_tensor, score = dataset[0]
    print(f"Sample 0 - Video tensor shape: {video_tensor.shape}")
    print(f"Sample 0 - Score: {score}")

    # Check that the tensor shapes are as expected
    assert video_tensor.shape == (3, 16, 224, 224)
    print("Shape assertion passed.")