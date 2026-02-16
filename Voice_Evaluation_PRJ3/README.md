# Voice Behavioral Analysis Module

## Overview

This module analyzes speech delivery behavior from interview audio.

It does NOT evaluate content.
It evaluates *how* something is said, not *what* is said.

The model is trained using ranking supervision from human-ranked interview videos.

---

## What This Module Does

1. Extracts acoustic features from audio
2. Learns a ranking-based stability representation
3. Produces:
   - Stability score (ML learned)
   - Pitch over time
   - Loudness over time
   - Words Per Minute (via STT integration)

This module is designed to integrate into a multimodal virtual interviewer system.

---

## Architecture

Audio Input
→ Feature Extraction
→ Temporal Model (BiLSTM + Attention)
→ Scalar Stability Score

Optional:
Audio → Speech-to-Text → WPM calculation

---

## Features Used

From each 25ms frame:

- RMS Energy (loudness)
- Zero Crossing Rate
- Spectral Centroid
- Spectral Bandwidth
- Pitch (F0 via autocorrelation)
- Voicing flag
- Pitch jitter
- Pause proxy

---

## Training Objective

The model is trained using pairwise ranking:

If sample A is ranked higher than sample B,
then model(A) > model(B).

Loss: Margin Ranking Loss  
Metric: Spearman Correlation

---

## Evaluation Outputs

Single analysis mode:
- Stability timeline
- Pitch timeline
- Loudness timeline
- Words per minute

Comparative mode:
- Stability comparison between two recordings

---

## Limitations

- Voice-only signal cannot fully explain interview performance.
- Stability is a learned behavioral representation, not a confidence score.
- Words per minute requires speech recognition.

---

## Integration in Main Project

Voice module provides one behavioral signal.

Final feedback is computed via multimodal fusion:

Voice + Face + Eye + Conversational flow
