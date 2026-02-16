import unittest
import torch
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.dataset.recruitview_dataset import RecruitViewDataset

class TestRecruitViewDataset(unittest.TestCase):

    METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata.jsonl"
    VIDEO_ROOT = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\videos"

    def test_dataset_initialization(self):
        """Test if the dataset can be initialized."""
        dataset = RecruitViewDataset(self.METADATA_PATH, self.VIDEO_ROOT)
        self.assertIsNotNone(dataset)

    def test_dataset_length(self):
        """Test if the dataset length is correct."""
        dataset = RecruitViewDataset(self.METADATA_PATH, self.VIDEO_ROOT)
        # From the inspection phase, we know there are 2011 records.
        self.assertEqual(len(dataset), 2011)

    def test_getitem(self):
        """Test if getting an item returns the correct types and shapes."""
        dataset = RecruitViewDataset(self.METADATA_PATH, self.VIDEO_ROOT, target_column='confidence_score')
        video_tensor, score = dataset[1] # Index 1 has a confidence score

        # Check types
        self.assertIsInstance(video_tensor, torch.Tensor)
        self.assertIsInstance(score, torch.Tensor)

        # Check video tensor shape (T, H, W, C)
        self.assertEqual(len(video_tensor.shape), 4)
        self.assertGreater(video_tensor.shape[0], 0) # T > 0
        self.assertGreater(video_tensor.shape[1], 0) # H > 0
        self.assertGreater(video_tensor.shape[2], 0) # W > 0
        self.assertEqual(video_tensor.shape[3], 3)   # C = 3

        # Check score shape (should be a scalar)
        self.assertEqual(score.shape, torch.Size([]))

if __name__ == '__main__':
    unittest.main()
