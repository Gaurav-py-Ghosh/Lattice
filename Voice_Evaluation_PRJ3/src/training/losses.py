import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, score_a, score_b):
        # score_a should be greater than score_b
        loss = torch.clamp(self.margin - (score_a - score_b), min=0.0)
        return loss.mean()
