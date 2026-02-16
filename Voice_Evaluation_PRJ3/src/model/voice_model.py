import torch
import torch.nn as nn


class VoiceRankingModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        x: [B, T, 4] or [T, 4]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, T, 4]

        lstm_out, _ = self.lstm(x)  # [B, T, 2H]

        attn_scores = self.attention(lstm_out)        # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, 1) # [B, T, 1]

        pooled = (lstm_out * attn_weights).sum(dim=1)  # [B, 2H]

        score = self.output(pooled)  # [B, 1]

        return score.squeeze(-1)     # [B]
if __name__ == "__main__":
    model = VoiceRankingModel()
    dummy = torch.randn(300, 4)
    out = model(dummy)
    print(out.shape)
