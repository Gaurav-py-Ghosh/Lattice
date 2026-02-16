import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioEncoder(nn.Module):
    """
    A wrapper for the pre-trained Wav2Vec2 model from Hugging Face.
    Input: A batch of raw audio waveforms
    Output: A sequence of features (B, T, D)
    """
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        # The output dimension of the base model's feature extractor is 768
        self.output_dim = 768

    def forward(self, x):
        # Wav2Vec2Model expects a 1D tensor or a batch of 1D tensors,
        # but our dataloader provides a padded 2D tensor.
        # We process each item in the batch individually for now.
        # Note: This is less efficient than a fully batched operation but handles variable lengths.
        # A more advanced implementation would use padding and an attention_mask.
        
        # Squeeze the channel dimension
        if x.dim() == 3:
            x = x.squeeze(1)

        outputs = self.wav2vec2(x)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

class AudioModel(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        encoder_output_dim = self.audio_encoder.output_dim # Should be 768

        # Temporal modeling stack (same as VideoModel)
        self.bilstm = nn.LSTM(
            input_size=encoder_output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=2 * hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.conv1d = nn.Conv1d(
            in_channels=2 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )

        self.query_vector = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, audio_waveform):
        # audio_waveform: (B, num_samples) - padded batch of waveforms

        # 1. Audio Encoder
        # Output: (B, T, D) where D is 768
        encoded_features = self.audio_encoder(audio_waveform)
        
        # 2. Temporal Modeling Stack
        bilstm_output, _ = self.bilstm(encoded_features)
        attn_output, _ = self.temporal_attention(query=bilstm_output, key=bilstm_output, value=bilstm_output)
        
        conv1d_input = attn_output.permute(0, 2, 1)
        conv1d_output = self.conv1d(conv1d_input)
        conv1d_output = conv1d_output.permute(0, 2, 1)

        # 3. Attention Pooling
        query = self.query_vector.expand(conv1d_output.size(0), -1, -1)
        pooled_output, _ = self.attention_pooling(query=query, key=conv1d_output, value=conv1d_output)
        pooled_output = pooled_output.squeeze(1)

        # 4. Output Head
        output_scalar = self.output_head(pooled_output)
        
        return output_scalar

if __name__ == '__main__':
    # Sanity check with dummy data
    batch_size = 2
    sample_rate = 16000
    sequence_length_seconds = 10
    num_samples = sample_rate * sequence_length_seconds

    dummy_audio_waveforms = torch.randn(batch_size, num_samples)

    model = AudioModel()
    output = model(dummy_audio_waveforms)

    print(f"Input shape: {dummy_audio_waveforms.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1), f"Expected output shape ({batch_size}, 1), but got {output.shape}"
    print("Shape assertions passed for AudioModel.")
