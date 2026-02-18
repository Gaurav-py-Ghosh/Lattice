# Voice Evaluation Project - WAV2VEC2 Based Speaking Skills Assessment

## Overview

This project implements a **deep learning-based voice evaluation system** that assesses speaking skills from audio recordings using a fine-tuned Wav2Vec2 model. The system can predict speaking skills scores, analyze voice characteristics over time, and compare multiple audio samples.

## Project Objective

Evaluate and rank candidates' **speaking skills** based on audio features extracted from interview recordings. The model is trained on a dataset of voice samples with corresponding speaking skills scores.

---

## Architecture & Components

### 1. **Core Model: `src/model/voice_wav2vec_model.py`**

- **Base Architecture**: Facebook's Wav2Vec2-Base (pre-trained on unlabeled speech)
- **Approach**: Transfer learning with frozen encoder layers for efficiency
- **Pipeline**:
  1. Raw audio input → Wav2Vec2 encoder
  2. Mean pooling over temporal dimension
  3. Multi-layer regression head → Speaking skills score
  
```
Input Audio [B, T] → Wav2Vec2 (768-dim hidden) → 
Pooling → Regression (512 → 256 → 1) → Score Output
```

**Key Features**:
- Leverages pre-trained speech representations
- Frozen Wav2Vec2 layers (parameter efficiency)
- Output: Score (continuous) + Embedding (256-dim)

---

### 2. **Dataset Module: `src/dataset/voice_wav_dataset.py`**

- **Input**: CSV with file names and ground truth speaking skills scores
- **Data Source**: RecruitView audio interviews
- **Processing**:
  - Loads WAV files at 16kHz sample rate
  - Handles mono/stereo conversion (averages stereo to mono)
  - Pads or trims to 15 seconds max
  - Returns normalized waveforms + labels

**Dataset Structure**:
```
CSV Columns: [file_name, speaking_skills, ...]
Audio Format: WAV, 16kHz, mono
```

---

### 3. **Training Pipeline: `src/training/train_voice_wav2vec.py`**

- **Loss Function**: Huber Loss (delta=1.0)
- **Optimizer**: Adam (learning rate: 1e-4)
- **Batch Size**: 4
- **Epochs**: 5
- **Device**: CUDA if available, else CPU

**Training Process**:
1. Load dataset from CSV + audio directory
2. Create DataLoader with custom collate function
3. Train model for specified epochs
4. Save trained model weights to `voice_wav2vec_model.pt`

---

### 4. **Audio Feature Extraction: `audio_features.py`**

Comprehensive audio feature extraction module for voice analysis:

**Extracted Features**:
1. **Energy (RMS)**: Loudness/intensity of speech
2. **Zero-Crossing Rate (ZCR)**: Approximate noise vs voiced distinction
3. **Spectral Centroid**: Center of mass in frequency domain
4. **Spectral Bandwidth**: Spread of energy in frequency
5. **Pitch (F0)**: Fundamental frequency via autocorrelation
6. **Voicing**: Binary indicator of voiced/unvoiced frames
7. **Pitch Jitter**: Pitch instability measure
8. **Pause Detection**: Identifies silent/pause regions

**Processing Pipeline**:
- Frame signal (400 samples/frame, 160 sample hop)
- Extract 8-dimensional feature vector per frame
- Smooth features for noise reduction

---

### 5. **Main Evaluation Script: `voice_evaluation_wav2vec.py`**

The central interface for voice evaluation with two analysis modes:

#### **Mode 1: Single Analysis**
- Input: Single audio file or live recording
- Output: 
  - Overall speaking skills score
  - Percentile ranking vs training dataset
  - 3-panel visualization:
    - Sliding window scores over time
    - Energy (loudness) profile
    - Pitch proxy profile

#### **Mode 2: Comparative Analysis**
- Input: Two audio files/recordings
- Output:
  - Individual scores for both inputs
  - Score difference
  - Overlaid comparison plot

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `load_model()` | Load pre-trained model weights |
| `load_dataset_scores()` | Load reference dataset scores for percentile calculation |
| `prepare_input()` | Convert various audio formats to WAV via ffmpeg |
| `record_audio()` | Live recording with sounddevice |
| `load_waveform()` | Load and normalize audio to model specs |
| `predict_score()` | Forward pass through model |
| `sliding_window_analysis()` | Temporal analysis with 5-sec windows, 2-sec stride |
| `extract_pitch_energy()` | Extract energy and pitch proxy features |
| `plot_single_report()` | Visualize single analysis |
| `plot_comparison_report()` | Visualize comparative analysis |

---

## Configuration & Constants

```python
MODEL_PATH = "voice_wav2vec_model.pt"           # Pre-trained model weights
CSV_PATH = "recruitview - Copy.csv"             # Dataset with labels
SAMPLE_RATE = 16000                             # Audio sample rate (Hz)
MAX_SECONDS = 15                                # Maximum audio duration
FRAME_SIZE = 400                                # Frame length for features
HOP_SIZE = 160                                  # Frame hop length
```

---

## Data Flow

```
User Input (Audio File/Recording)
    ↓
[prepare_input] → Convert to WAV if needed
    ↓
[load_waveform] → Normalize to 16kHz, 15-sec max
    ↓
[Model Inference] → Wav2Vec2 → Regression → Score
    ↓
[Percentile Calculation] → Compare vs dataset
    ↓
[Sliding Window Analysis] → Temporal breakdown
    ↓
[Feature Extraction] → Energy, Pitch, ZCR
    ↓
[Visualization] → Multi-panel plots
```

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running the Evaluation

```bash
python voice_evaluation_wav2vec.py
```

**Interactive Flow**:
1. Select analysis type (1=Single, 2=Comparative)
2. Select input mode (1=File, 2=Live recording)
3. Provide audio path(s) or start recording
4. View results and visualizations

**Example Commands**:

```bash
# Single file analysis
Enter analysis type: 1
Enter input mode: 1
Enter full path: C:\path\to\audio.wav

# Live recording comparison
Enter analysis type: 2
Enter input mode: 2
# Will prompt for two 15-second recordings
```

### Training a New Model

```bash
cd src/training
python train_voice_wav2vec.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `transformers` | Wav2Vec2 pre-trained model |
| `numpy` | Numerical computation |
| `scipy` | Signal processing, percentiles |
| `soundfile` | WAV file I/O |
| `sounddevice` | Live audio recording |
| `matplotlib` | Visualization |
| `ffmpeg-python` | Audio format conversion |
| `tqdm` | Progress bars |

---

## Output Files Generated

| File | Purpose |
|------|---------|
| `voice_wav2vec_model.pt` | Trained model weights |
| `converted_*.wav` | Temporary converted audio files |
| `recorded_*.wav` | Recorded audio samples |
| Plots | PNG visualizations (displayed via matplotlib) |

---

## Model Details

**Pre-trained Base**: 
- facebook/wav2vec2-base
- 768-dimensional hidden representations
- Self-supervised training on LibriSpeech unlabeled data

**Fine-tuned Head**:
- Adapts frozen Wav2Vec2 features to speaking skills prediction
- Small trainable MLP for efficiency and speed

**Output**: 
- Speaking skills score (continuous, typically 0-100 range)
- 256-dimensional embedding for feature analysis

---

## Results & Evaluation

The system provides:

1. **Absolute Scores**: Direct speaking skills assessment (0-100)
2. **Relative Ranking**: Percentile comparison vs training dataset
3. **Temporal Analysis**: Score variation across 5-second windows
4. **Voice Characteristics**: Energy and pitch profiles
5. **Comparative Metrics**: Head-to-head score comparison

---

## File Structure

```
Voice_Evaluation_PRJ3/
├── voice_evaluation_wav2vec.py     # Main evaluation script
├── audio_features.py               # Feature extraction utilities
├── voice_wav2vec_model.pt          # Pre-trained model weights
├── recruitview - Copy.csv          # Training dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── feature_cache/              # Cached audio features (1100+ files)
└── src/
    ├── model/
    │   ├── voice_wav2vec_model.py  # Model architecture
    │   └── voice_modelcopy.py      # Backup
    ├── dataset/
    │   └── voice_wav_dataset.py    # Dataset loader
    └── training/
        ├── train_voice_wav2vec.py   # Training script
        ├── validate_voice_wav2vec.py # Validation
        └── *_rankercopy.py          # Alternative models

