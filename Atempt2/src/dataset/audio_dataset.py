import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm

class RecruitViewAudioDataset(Dataset):
    """
    PyTorch Dataset for the RecruitView dataset, specializing in audio data.
    Loads audio files, applies transformations, and returns a single target score.
    """
    def __init__(self, metadata_path, audio_root, target_column='speaking_skills', sample_rate=16000):
        """
        Args:
            metadata_path (str): Path to the metadata_with_audio.jsonl file.
            audio_root (str): Path to the directory containing the audio files.
            target_column (str): The name of the target column to use.
            sample_rate (int): The target sample rate for audio.
        """
        self.audio_root = audio_root
        self.target_column = target_column
        self.sample_rate = sample_rate
        
        # Load metadata with audio paths
        print(f"Loading metadata from: {metadata_path}")
        data = []
        with open(metadata_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.metadata = pd.DataFrame(data)
        
        # Handle NaN values in the target column
        self.metadata[self.target_column] = self.metadata[self.target_column].fillna(0.0)

        # Filter out records where audio_path might be missing if extraction failed for some
        initial_len = len(self.metadata)
        self.metadata = self.metadata.dropna(subset=['audio_path'])
        if len(self.metadata) < initial_len:
            print(f"Warning: Filtered out {initial_len - len(self.metadata)} records due to missing audio_path.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (audio_waveform, target_score)
                   audio_waveform is of shape (1, num_samples)
        """
        audio_info = self.metadata.iloc[idx]
        
        audio_path = audio_info['audio_path']

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Resample if necessary
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a dummy tensor and target if audio is corrupted or missing
            waveform = torch.zeros((1, self.sample_rate * 5)) # 5 seconds of silent audio
            
        # Get target score
        target_score = torch.tensor(audio_info[self.target_column], dtype=torch.float32)
        
        return waveform, target_score

if __name__ == '__main__':
    # Example usage and sanity check
    # Make sure to replace with your actual paths
    METADATA_WITH_AUDIO_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata_with_audio.jsonl"
    AUDIO_ROOT_DIR = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\audio"

    # Create dataset for speaking_skills
    dataset = RecruitViewAudioDataset(METADATA_WITH_AUDIO_PATH, AUDIO_ROOT_DIR, target_column='speaking_skills')
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    audio_waveform, score = dataset[0]
    print(f"Sample 0 - Audio waveform shape: {audio_waveform.shape}")
    print(f"Sample 0 - Score: {score}")

    # The waveform shape will vary based on duration, so just check it's a 2D tensor
    assert audio_waveform.dim() == 2 and audio_waveform.shape[0] == 1, \
        f"Expected audio waveform shape (1, num_samples), but got {audio_waveform.shape}"
    print("Shape assertion passed for RecruitViewAudioDataset.")
