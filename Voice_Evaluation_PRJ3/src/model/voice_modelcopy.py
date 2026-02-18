import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceRankingModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, embed_dim=256):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        # --- CRMF-lite Experts ---
        self.expert1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        self.expert2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        self.expert3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        # Router
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Final regression head
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(0)

        lstm_out, _ = self.lstm(x)

        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, 1)

        pooled = (lstm_out * attn_weights).sum(dim=1)

        # Experts
        e1 = self.expert1(pooled)
        e2 = self.expert2(pooled)
        e3 = self.expert3(pooled)

        # Router weights
        weights = torch.softmax(self.router(pooled), dim=1)

        # Weighted fusion
        fused = (
            weights[:, 0:1] * e1 +
            weights[:, 1:2] * e2 +
            weights[:, 2:3] * e3
        )

        score = self.regressor(fused)

        return score.squeeze(-1), fused

if __name__ == "__main__":
    model = VoiceRankingModel(input_dim=8)
    dummy = torch.randn(300, 8)
    score, embedding = model(dummy)
    print("Score shape:", score.shape)
    print("Embedding shape:", embedding.shape)

