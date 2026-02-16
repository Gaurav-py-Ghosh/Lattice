import torch
import torch.nn as nn
from scipy.stats import spearmanr
import unittest
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


class PairwiseRankingLoss(nn.Module):
    def forward(self, preds, targets):
        # The loss is undefined for a batch size of less than 2
        if preds.shape[0] < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        preds = preds.squeeze()
        targets = targets.squeeze()

        # Handle the case where squeeze() might result in a 0-dim tensor
        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

        # Pairwise differences
        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        # Keep only ordered pairs (where target j > target i)
        mask = target_diff > 0

        # Logistic ranking loss
        loss = torch.log1p(torch.exp(-pred_diff[mask]))
        
        # Handle the case where there are no valid pairs in the batch
        if loss.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        return loss.mean()


class HuberLoss(nn.SmoothL1Loss):
    """
    Huber Loss. Wraps nn.SmoothL1Loss.
    """
    def __init__(self, reduction='mean', beta=1.0):
        super().__init__(reduction=reduction, beta=beta)

# Unit tests for the losses
class TestLossFunctions(unittest.TestCase):

    def test_pairwise_ranking_loss_logic(self):
        """Tests that incorrectly ordered preds have higher loss than correctly ordered ones."""
        targets = torch.tensor([10.0, 20.0, 30.0])
        # Predictions that have the same ranking as targets
        correct_preds = torch.tensor([1.0, 2.0, 3.0])
        # Predictions that have the opposite ranking of targets
        incorrect_preds = torch.tensor([3.0, 2.0, 1.0])
        
        loss_fn = PairwiseRankingLoss()
        
        loss_correct = loss_fn(correct_preds, targets)
        loss_incorrect = loss_fn(incorrect_preds, targets)
        
        # The core property: incorrect order should have a higher loss
        self.assertGreater(loss_incorrect.item(), loss_correct.item())

    def test_pairwise_ranking_loss_no_valid_pairs(self):
        """Tests that loss is 0 when no pairs can be formed for ranking."""
        # All targets are the same, so no target_diff > 0
        targets = torch.tensor([10.0, 10.0, 10.0])
        preds = torch.tensor([1.0, 2.0, 3.0])
        
        loss_fn = PairwiseRankingLoss()
        loss = loss_fn(preds, targets)
        
        # The guard clause should catch this and return 0.0
        self.assertAlmostEqual(loss.item(), 0.0, places=4)
        
    def test_pairwise_ranking_loss_batch_size_one(self):
        """Tests that loss is 0 for a batch of size 1."""
        targets = torch.tensor([10.0])
        preds = torch.tensor([1.0])
        
        loss_fn = PairwiseRankingLoss()
        loss = loss_fn(preds, targets)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_huber_loss_basic(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = HuberLoss()(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0)

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = HuberLoss()(pred, target)
        # With default beta=1.0, for |x-y| <= 1, loss is 0.5 * (x-y)^2.
        # Here, all diffs are exactly 1, so the loss per item is 0.5 * 1^2 = 0.5.
        # The mean of these is 0.5.
        self.assertAlmostEqual(loss.item(), 0.5) # Corrected expected value

if __name__ == '__main__':
    unittest.main()