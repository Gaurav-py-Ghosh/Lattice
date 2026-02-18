import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class VoiceWav2VecModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # Pretrained Wav2Vec2 Base
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # Freeze most layers for stability & speed
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        hidden_size = self.wav2vec.config.hidden_size  # 768

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

        self.output = nn.Linear(embed_dim, 1)

    def forward(self, input_values):

        # input_values: [B, T]
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state  # [B, T', 768]

        # Mean pooling over time
        pooled = hidden_states.mean(dim=1)  # [B, 768]

        embedding = self.regressor(pooled)  # [B, embed_dim]
        score = self.output(embedding)      # [B, 1]

        return score.squeeze(-1), embedding
