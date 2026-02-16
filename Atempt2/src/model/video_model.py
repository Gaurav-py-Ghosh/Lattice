import torch
import torch.nn as nn
from transformers import AutoModelForVideoClassification

class VideoMAEEncoder(nn.Module):
    """
    A wrapper for the pre-trained VideoMAE model from Hugging Face.
    Input: (B, T, C, H, W) - Batch, Time, Channels, Height, Width
    Output: (B, T, D) - Batch, Time, Feature_Dimension (768 for base model)
    """
    def __init__(self):
        super().__init__()
        # Load the pre-trained VideoMAE model, it directly contains the backbone
        # We need to access its base model to get the feature extractor
        self.videomae = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").videomae
        # The output dimension of the base model is 768
        self.output_dim = 768

    def forward(self, x):
        # The input x is now expected to be in (B, C, T, H, W) format
        outputs = self.videomae(x)
        # The last_hidden_state has shape (B, T, D)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

class VideoModel(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.video_encoder = VideoMAEEncoder()
        encoder_output_dim = self.video_encoder.output_dim # Should be 768

        # Temporal modeling stack
        # 1. BiLSTM
        self.bilstm = nn.LSTM(
            input_size=encoder_output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Temporal Self-Attention
        # The input to attention will be the concatenated output of BiLSTM (2 * hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=2 * hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 3. Conv1D
        self.conv1d = nn.Conv1d(
            in_channels=2 * hidden_dim, # From BiLSTM or attention
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        # Attention pooling (to get a fixed-size representation from variable-length sequence)
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_dim, # Output of Conv1D
            num_heads=1,
            batch_first=True
        )

        # Learnable query vector for attention pooling
        self.query_vector = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # One output head -> one scalar (no activation)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, video_frames):
        # video_frames: (B, C, T, H, W)

        # 1. Video Encoder
        # Output: (B, T, D) where D is 768
        encoded_features = self.video_encoder(video_frames)
        
        # 2. Temporal Modeling Stack
        # BiLSTM
        bilstm_output, _ = self.bilstm(encoded_features) # (B, T, 2 * hidden_dim)

        # Temporal Self-Attention
        attn_output, _ = self.temporal_attention(
            query=bilstm_output,
            key=bilstm_output,
            value=bilstm_output
        ) # (B, T, 2 * hidden_dim)

        # Conv1D
        # Input to Conv1D needs to be (B, C, T)
        conv1d_input = attn_output.permute(0, 2, 1) # (B, 2 * hidden_dim, T)
        conv1d_output = self.conv1d(conv1d_input)   # (B, hidden_dim, T)
        conv1d_output = conv1d_output.permute(0, 2, 1) # (B, T, hidden_dim)

        # 3. Attention Pooling
        # Expand the learnable query vector to the batch size
        query = self.query_vector.expand(conv1d_output.size(0), -1, -1) # (B, 1, hidden_dim)

        pooled_output, _ = self.attention_pooling(
            query=query,
            key=conv1d_output,
            value=conv1d_output
        ) # (B, 1, hidden_dim)
        
        # Remove the sequence dimension of 1
        pooled_output = pooled_output.squeeze(1) # (B, hidden_dim)

        # 4. Output Head (one scalar, no activation)
        output_scalar = self.output_head(pooled_output) # (B, 1)

        return output_scalar

if __name__ == '__main__':
    # Sanity check with dummy data
    batch_size = 2
    sequence_length = 16 # Number of frames for VideoMAE
    channels = 3
    height = 224
    width = 224

    dummy_video_frames = torch.randn(batch_size, sequence_length, channels, height, width)

    model = VideoModel()
    output = model(dummy_video_frames)

    print(f"Input shape: {dummy_video_frames.shape}")
    print(f"Output shape: {output.shape}")
    
    # Assertions for shape
    assert output.shape == (batch_size, 1), f"Expected output shape ({batch_size}, 1), but got {output.shape}"
    print("Shape assertions passed for VideoModel.")
