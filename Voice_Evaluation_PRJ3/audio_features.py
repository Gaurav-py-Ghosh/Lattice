import numpy as np
import torch
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_SIZE = 400    
HOP_SIZE = 160      


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    assert sr == SAMPLE_RATE
    return torch.from_numpy(wav)

def frame_signal(wav):
    frames = []
    for i in range(0, len(wav) - FRAME_SIZE, HOP_SIZE):
        frames.append(wav[i:i+FRAME_SIZE])
    return torch.stack(frames)


def rms_energy(frames):
    return torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)

def zero_crossing_rate(frames):
    signs = torch.sign(frames)
    return torch.mean((signs[:, 1:] != signs[:, :-1]).float(), dim=1)

def spectral_features(frames):
    fft = torch.fft.rfft(frames, dim=1)
    mag = torch.abs(fft)
    freqs = torch.linspace(0, SAMPLE_RATE / 2, mag.shape[1])
    centroid = (mag * freqs).sum(dim=1) / (mag.sum(dim=1) + 1e-8)
    bandwidth = torch.sqrt(
        ((freqs - centroid.unsqueeze(1)) ** 2 * mag).sum(dim=1) /
        (mag.sum(dim=1) + 1e-8)
    )
    return centroid, bandwidth

def autocorr_pitch(frame, fmin=80, fmax=300):
    x = frame - frame.mean()
    if torch.all(x == 0):
        return 0.0, 0.0

    fft = torch.fft.rfft(x, n=2 * x.numel())
    ac = torch.fft.irfft(fft * torch.conj(fft))[:x.numel()]
    ac = ac / (ac[0] + 1e-8)

    min_lag = int(SAMPLE_RATE / fmax)
    max_lag = int(SAMPLE_RATE / fmin)
    if max_lag >= ac.numel():
        return 0.0, 0.0

    lag = torch.argmax(ac[min_lag:max_lag]) + min_lag
    voiced = 1.0 if ac[lag] > 0.3 else 0.0
    f0 = SAMPLE_RATE / lag if voiced else 0.0
    return f0, voiced

def pitch_and_voicing(frames):
    f0s, voiced = [], []
    for i in range(frames.shape[0]):
        f0, v = autocorr_pitch(frames[i])
        f0s.append(f0)
        voiced.append(v)
    return torch.tensor(f0s), torch.tensor(voiced)

def pitch_jitter(f0, voiced, win=5):
    jitter = torch.zeros_like(f0)
    for i in range(len(f0)):
        if voiced[i] == 0:
            continue
        lo = max(0, i - win)
        hi = min(len(f0), i + win + 1)
        vals = f0[lo:hi]
        vals = vals[vals > 0]
        jitter[i] = vals.std() if len(vals) > 1 else 0.0
    return jitter

def smooth(x, k=5):
    x_np = x.numpy()
    smoothed = np.convolve(x_np, np.ones(k) / k, mode="same")
    return torch.from_numpy(smoothed).float()

def extract_features(path):
    wav = load_audio(path)
    frames = frame_signal(wav)

    energy = rms_energy(frames)
    zcr = zero_crossing_rate(frames)
    centroid, bandwidth = spectral_features(frames)

    f0, voiced = pitch_and_voicing(frames)

    #  conditioning
    f0 = f0 * voiced                    
    f0 = torch.log(f0 + 1e-6)          
    f0 = smooth(f0)                     

    jitter = pitch_jitter(f0, voiced)
    jitter = smooth(jitter)

    pause = ((energy < energy.median()) & (voiced == 0)).float()

    features = torch.stack(
        [energy, zcr, centroid, bandwidth, f0, voiced, jitter, pause],
        dim=1
    )

    return features 

if __name__ == "__main__":
    feats = extract_features("data/audio/vid_1543.wav")
    print("Feature shape:", feats.shape)
