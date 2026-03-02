"""
Vision Gaze Tracking WebSocket Server

This server manages vision.py sessions and communicates with the Next.js frontend.
- Accepts uploaded video files via POST /upload_video
- Starts/stops vision.py subprocess in offline batch mode (--video required)
- Starts/stops realtime_inference.py for VideoMAE confidence prediction
- Manages session-specific log files
- Sends gaze data and confidence predictions to frontend via WebSocket

New model (offline/batch):
  Frontend uploads video  →  POST /upload_video  →  receives video_path
  Frontend sends start_session(video_path)  →  FastAPI launches vision.py
  vision.py processes video offline  →  exits at EOF
"""

import asyncio
import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor

# ---------------------------------------------------------------------------
# Voice module path — add Voice_Evaluation_PRJ3 to sys.path so we can import
# the VoiceWav2VecModel class without installing it as a package
# ---------------------------------------------------------------------------
_VOICE_ROOT = Path(__file__).parent.parent / "Voice_Evaluation_PRJ3"
if str(_VOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VOICE_ROOT))

try:
    from src.model.voice_wav2vec_model import VoiceWav2VecModel
    _VOICE_MODEL_CLASS_AVAILABLE = True
except ImportError as _e:
    _VOICE_MODEL_CLASS_AVAILABLE = False
    print(f"[VoiceAnalyzer] WARNING: could not import VoiceWav2VecModel: {_e}")

VOICE_MODEL_PATH = str(_VOICE_ROOT / "voice_wav2vec_model.pt")
VOICE_SAMPLE_RATE = 16000
VOICE_MAX_SECONDS = 15

# app is created after lifespan is defined below

# ---------------------------------------------------------------------------
# VoiceAnalyzer — wraps the wav2vec speaking-skills model
# ---------------------------------------------------------------------------

class VoiceAnalyzer:
    """Wrapper around VoiceWav2VecModel.

    Model is loaded ONCE at server startup (eager) and shared across all
    concurrent chunk requests. A threading.Lock prevents duplicate loads
    if startup somehow races.
    """

    def __init__(self):
        self._model = None
        self._device = None
        self._lock = threading.Lock()  # prevents concurrent load attempts

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self):
        """Load model — safe to call multiple times; only loads once."""
        if self._model is not None:  # fast path, no lock needed
            return True
        with self._lock:
            if self._model is not None:  # second check inside lock
                return True
            if not _VOICE_MODEL_CLASS_AVAILABLE:
                print("[VoiceAnalyzer] Model class unavailable — skipping load")
                return False
            if not os.path.exists(VOICE_MODEL_PATH):
                print(f"[VoiceAnalyzer] Model file not found: {VOICE_MODEL_PATH}")
                return False
            try:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[VoiceAnalyzer] Loading model on {self._device}…")
                self._model = VoiceWav2VecModel().to(self._device)
                self._model.load_state_dict(
                    torch.load(VOICE_MODEL_PATH, map_location=self._device, weights_only=False)
                )
                self._model.eval()
                print(f"[VoiceAnalyzer] Model ready on {self._device}")
                return True
            except Exception as e:
                print(f"[VoiceAnalyzer] ERROR loading model: {e}")
                return False

    # keep _load as an alias so nothing else breaks
    _load = load

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_audio(input_path: str, output_wav: str) -> str:
        """Use ffmpeg to extract/convert audio to 16 kHz mono WAV."""
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", str(VOICE_SAMPLE_RATE),
            output_wav
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for: {input_path}")
        return output_wav

    @staticmethod
    def _load_waveform(path: str) -> torch.Tensor:
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        max_len = VOICE_SAMPLE_RATE * VOICE_MAX_SECONDS
        if len(wav) > max_len:
            wav = wav[:max_len]
        else:
            wav = np.pad(wav, (0, max_len - len(wav)))
        return torch.tensor(wav)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _predict_score(self, waveform: torch.Tensor) -> float:
        waveform = waveform.unsqueeze(0).to(self._device)
        with torch.no_grad():
            score, _ = self._model(waveform)
        return score.item()

    def _sliding_window(self, waveform: torch.Tensor,
                        window_sec: int = 5,
                        stride_sec: int = 2):
        window_len = VOICE_SAMPLE_RATE * window_sec
        stride_len = VOICE_SAMPLE_RATE * stride_sec
        scores, times = [], []
        for start in range(0, len(waveform) - window_len, stride_len):
            segment = waveform[start:start + window_len].unsqueeze(0).to(self._device)
            with torch.no_grad():
                score, _ = self._model(segment)
            scores.append(score.item())
            times.append(start / VOICE_SAMPLE_RATE)
        return np.array(times), np.array(scores)

    @staticmethod
    def _extract_features(wav_path: str):
        wav, _ = sf.read(wav_path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        energy = np.sqrt(
            np.convolve(wav ** 2, np.ones(400) / 400, mode="same")
        ).tolist()
        zcr = np.abs(np.diff(np.sign(wav))).astype(float).tolist()
        return energy, zcr

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, input_path: str, session_id: str = "tmp") -> dict:
        """Run full voice analysis on a video or audio file.

        Returns a JSON-serialisable dict:
        {
            "score": float,
            "window_times": [...],
            "window_scores": [...],
            "energy": [...],
            "pitch_proxy": [...],
            "error": str | None
        }
        """
        if not self._load():
            return {"error": "Voice model not available", "score": None}

        try:
            ext = os.path.splitext(input_path)[1].lower()
            wav_path = input_path if ext == ".wav" else self._extract_audio(
                input_path,
                str(Path(input_path).parent / f"voice_tmp_{session_id}.wav")
            )

            waveform = self._load_waveform(wav_path)
            score = self._predict_score(waveform)
            times, window_scores = self._sliding_window(waveform)
            energy, pitch_proxy = self._extract_features(wav_path)

            return {
                "score": round(score, 4),
                "window_times": times.tolist(),
                "window_scores": window_scores.tolist(),
                "energy": energy,
                "pitch_proxy": pitch_proxy,
                "error": None
            }
        except Exception as e:
            print(f"[VoiceAnalyzer] ERROR during analysis: {e}")
            return {"error": str(e), "score": None}


# Single shared instance
voice_analyzer = VoiceAnalyzer()

# ---------------------------------------------------------------------------
# VideoInferenceAnalyzer — in-process VideoMAE confidence model
# ---------------------------------------------------------------------------

_ATEMPT2_ROOT = Path(__file__).parent.parent / "Atempt2"
if str(_ATEMPT2_ROOT) not in sys.path:
    sys.path.insert(0, str(_ATEMPT2_ROOT))

VIDEO_MODEL_PATH = str(_ATEMPT2_ROOT / "checkpoints" / "videoMAE_confidence_ranker_epoch6.pth")
FACIAL_MODEL_PATH = str(_ATEMPT2_ROOT / "checkpoints" / "best_facial_expression_model.pth")
VIDEO_NUM_FRAMES = 16

try:
    from src.model.video_model import VideoModel as _VideoModel
    _VIDEO_MODEL_CLASS_AVAILABLE = True
except ImportError as _ve:
    _VIDEO_MODEL_CLASS_AVAILABLE = False
    print(f"[VideoAnalyzer] WARNING: could not import VideoModel: {_ve}")


class _BaseVideoAnalyzer:
    """Shared logic for in-process VideoMAE-based analyzers."""

    def __init__(self, label: str, model_path: str):
        self._label = label
        self._model_path = model_path
        self._model = None
        self._processor = None
        self._device = None
        self._lock = threading.Lock()

    def load(self):
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if not _VIDEO_MODEL_CLASS_AVAILABLE:
                print(f"[{self._label}] VideoModel class unavailable — skipping load")
                return False
            if not os.path.exists(self._model_path):
                print(f"[{self._label}] Model file not found: {self._model_path}")
                return False
            try:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[{self._label}] Loading model on {self._device}…")
                self._processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
                self._model = _VideoModel().to(self._device)
                ckpt = torch.load(self._model_path, map_location=self._device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                self._model.load_state_dict(state)
                self._model.eval()
                print(f"[{self._label}] Model ready on {self._device}")
                return True
            except Exception as e:
                print(f"[{self._label}] ERROR loading model: {e}")
                return False

    def _extract_frames(self, video_path: str, num_frames: int = VIDEO_NUM_FRAMES) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from: {video_path}")
        # Pad if short
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return np.array(frames[:num_frames])

    def _run_inference(self, video_path: str) -> float:
        frames = self._extract_frames(video_path)
        frames_list = [frame for frame in frames]
        inputs = self._processor(frames_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            output = self._model(pixel_values)
        return float(output.item())

    def analyze(self, video_path: str, session_id: str = "tmp") -> dict:
        if not self.load():
            return {"score": None, "error": f"{self._label} model not available"}
        try:
            score = self._run_inference(video_path)
            return {"score": round(score, 4), "error": None}
        except Exception as e:
            print(f"[{self._label}] ERROR: {e}")
            return {"score": None, "error": str(e)}


class VideoInferenceAnalyzer(_BaseVideoAnalyzer):
    def __init__(self):
        super().__init__("VideoMAE", VIDEO_MODEL_PATH)


class FacialExpressionAnalyzer(_BaseVideoAnalyzer):
    def __init__(self):
        super().__init__("FacialExpression", FACIAL_MODEL_PATH)


video_inference_analyzer = VideoInferenceAnalyzer()
facial_expression_analyzer = FacialExpressionAnalyzer()

# ---------------------------------------------------------------------------
# FastAPI lifespan — eager-load all three models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Load all ML models in thread-pool workers at startup."""
    global CHUNK_SEMAPHORE
    CHUNK_SEMAPHORE = asyncio.Semaphore(3)
    loop = asyncio.get_event_loop()
    print("[Startup] Pre-loading all models…")
    await asyncio.gather(
        loop.run_in_executor(None, voice_analyzer.load),
        loop.run_in_executor(None, video_inference_analyzer.load),
        loop.run_in_executor(None, facial_expression_analyzer.load),
    )
    print("[Startup] All models ready.")
    # Check ffmpeg availability (required for voice analysis audio extraction)
    _ffmpeg_exe = os.environ.get("FFMPEG_PATH", "ffmpeg")
    try:
        subprocess.run(
            [_ffmpeg_exe, "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        print(f"[Startup] ffmpeg OK ({_ffmpeg_exe})")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            f"[Startup] WARNING: ffmpeg not found at '{_ffmpeg_exe}'. "
            "Voice analysis will fail for video chunks. "
            "Install ffmpeg, add it to PATH, or set FFMPEG_PATH env variable."
        )
    yield  # server runs here

# Create app here so we can pass lifespan
app = FastAPI(lifespan=lifespan)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Store active sessions
# ---------------------------------------------------------------------------
active_sessions = {}

# Max concurrent chunk-processing tasks (each spawns a vision.py subprocess).
# Initialised in lifespan so it's bound to the correct event loop.
CHUNK_SEMAPHORE: asyncio.Semaphore  # forward declaration; assigned in lifespan


class VisionSession:
    def __init__(self, session_id: str,
                 video_path: str,
                 loop: bool = False,
                 speed: float = 1.0):
        self.session_id = session_id
        self.headless = True  # Always headless: offline batch mode, no OpenCV windows
        self.video_path = video_path  # Path to uploaded video file (required)
        self.loop = loop
        self.speed = speed
        self.vision_process: Optional[subprocess.Popen] = None
        self.inference_process: Optional[subprocess.Popen] = None
        self.log_file_path = Path(__file__).parent / "data" / f"gaze_log_{session_id}.txt"
        self.predictions_file = Path(__file__).parent / "data" / "predictions" / f"{session_id}_predictions.json"
        self.vision_log = Path(__file__).parent / "data" / f"vision_output_{session_id}.log"
        self.inference_log = Path(__file__).parent / "data" / f"inference_output_{session_id}.log"
        self.start_time = datetime.now()
        self.is_running = False
        self.last_prediction_count = 0
        
    def start(self):
        """Start vision.py and realtime_inference.py subprocesses for offline batch processing"""
        # Ensure data directory exists
        self.log_file_path.parent.mkdir(exist_ok=True)
        self.predictions_file.parent.mkdir(exist_ok=True)

        # video_path is required — no webcam fallback
        if not self.video_path:
            print("ERROR: video_path is required — no webcam fallback")
            self.is_running = False
            return
        if not os.path.exists(self.video_path):
            print(f"ERROR: Video file not found: {self.video_path}")
            self.is_running = False
            return

        # Use sys.executable to ensure we use the same Python (from conda env)
        python_exe = sys.executable
        vision_script = Path(__file__).parent / "vision.py"
        inference_script = Path(__file__).parent / "realtime_inference.py"

        print(f"Starting vision system with:")
        print(f"   Python: {python_exe}")
        print(f"   Vision Script: {vision_script}")
        print(f"   Inference Script: {inference_script}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Mode: offline batch (headless)")
        print(f"   Video file: {self.video_path}")
        print(f"   Loop: {self.loop} | Speed: {self.speed}x")

        # Build command arguments for vision.py
        # Always headless; --video is required (no webcam)
        vision_cmd = [
            python_exe, str(vision_script),
            "--session-id", self.session_id,
            "--headless",
            "--video", self.video_path,
        ]
        if self.loop:
            vision_cmd.append("--loop")
        if self.speed != 1.0:
            vision_cmd.extend(["--speed", str(self.speed)])
        
        # Build command arguments for inference.py
        inference_cmd = [
            python_exe, str(inference_script),
            "--session-id", self.session_id
        ]
        
        # Start vision.py with output logging
        try:
            self._vision_log_file = open(self.vision_log, 'w', encoding='utf-8')
            # PYTHONIOENCODING ensures emoji/unicode in vision.py print() go to the log cleanly
            child_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            if os.name == 'nt':  # Windows
                self.vision_process = subprocess.Popen(
                    vision_cmd,
                    cwd=Path(__file__).parent,
                    stdout=self._vision_log_file,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Linux/Mac
                self.vision_process = subprocess.Popen(
                    vision_cmd,
                    cwd=Path(__file__).parent,
                    stdout=self._vision_log_file,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )
            print(f"Started vision.py (PID: {self.vision_process.pid})")
            print(f"   Output log: {self.vision_log}")
        except Exception as e:
            print(f"ERROR: starting vision.py: {e}")
            self.is_running = False
            return
        
        self.is_running = True
        
        # Check if processes are still running
        time.sleep(0.5)
        if self.vision_process.poll() is not None:
            print(f"ERROR: vision.py crashed immediately (exit code: {self.vision_process.returncode})")
            self.is_running = False
            return
        
        print(f"Session started: {self.session_id}")
        print(f"   Log file: {self.log_file_path}")
        
    def stop(self):
        """Stop vision.py and realtime_inference.py subprocesses gracefully"""
        # Stop vision.py first
        if self.vision_process:
            try:
                # Try graceful shutdown first
                self.vision_process.terminate()
                self.vision_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if doesn't terminate
                self.vision_process.kill()
                self.vision_process.wait()
            finally:
                # Close the stdout log file handle to prevent leaking the fd
                if getattr(self, '_vision_log_file', None):
                    try:
                        self._vision_log_file.close()
                    except Exception:
                        pass
                    self._vision_log_file = None
            
            print(f"Stopped vision.py")
        
        # Stop inference.py (it will detect stop signal and finish)
        if self.inference_process:
            try:
                # Inference process will exit on its own when it detects stop signal
                # Give it time to finish processing
                self.inference_process.wait(timeout=10)
                print(f"Inference process completed naturally")
            except subprocess.TimeoutExpired:
                # If still running, terminate it
                self.inference_process.terminate()
                try:
                    self.inference_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.inference_process.kill()
                    self.inference_process.wait()
                print(f"Stopped realtime_inference.py")
        
        self.is_running = False
        print(f"Stopped session: {self.session_id}")
            
    def get_log_data(self) -> list:
        """Read and return log data from this session"""
        if not self.log_file_path.exists():
            return []
        
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse log entries
        log_entries = []
        for line in lines:
            line = line.strip()
            if ': ' in line:
                timestamp_str, status = line.split(': ', 1)
                log_entries.append({
                    'timestamp': timestamp_str,
                    'status': status
                })
        
        return log_entries
    
    def get_predictions(self) -> dict:
        """Read and return prediction data from this session"""
        if not self.predictions_file.exists():
            return {'predictions': [], 'summary': {}}
        
        try:
            with open(self.predictions_file, 'r') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError):
            return {'predictions': [], 'summary': {}}
    
    def get_new_predictions(self) -> list:
        """Get predictions that haven't been sent yet"""
        data = self.get_predictions()
        predictions = data.get('predictions', [])
        
        # Get only new predictions since last check
        new_predictions = predictions[self.last_prediction_count:]
        self.last_prediction_count = len(predictions)
        
        return new_predictions
    
    def cleanup(self):
        """Clean up session resources"""
        self.stop()


# ---------------------------------------------------------------------------
# Chunk processing — called for each 15-s video chunk from the frontend
# ---------------------------------------------------------------------------

async def _process_chunk(
    websocket: WebSocket,
    chunk_id: str,
    chunk_index: int,
    video_path: str,
) -> None:
    """Process a single 15-s video chunk through all three analysis modules:

    1. vision.py              — gaze tracking (subprocess, exits at EOF)
    2. video_inference_analyzer — VideoMAE confidence (in-process)
    3. facial_expression_analyzer — facial expression score (in-process)
    4. voice_analyzer         — speaking skills (in-process)

    All three ML models run concurrently once vision.py exits.
    Semaphore limits to CHUNK_SEMAPHORE concurrent chunks.
    Files are cleaned up in the finally block.
    """
    loop = asyncio.get_event_loop()

    async with CHUNK_SEMAPHORE:
        # ---- Create a mini VisionSession for gaze tracking only ----
        session = VisionSession(chunk_id, video_path=video_path)

        # Run session.start() in executor — it contains time.sleep() calls
        # that would otherwise block the event loop
        await loop.run_in_executor(None, session.start)

        if not session.is_running:
            try:
                await websocket.send_json({
                    'type': 'chunk_error',
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'message': f'Failed to start gaze session for chunk {chunk_id}',
                })
            except Exception:
                pass
            # Clean up the uploaded video even on early failure
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception:
                pass
            return

        try:
            # ---- Wait for vision.py to finish (it exits at EOF) ----
            await loop.run_in_executor(None, session.vision_process.wait)
            print(f"[Chunk {chunk_index}] vision.py finished (PID {session.vision_process.pid})")

            # ---- Run all three ML analyses concurrently ----
            voice_task = loop.run_in_executor(
                None, voice_analyzer.analyze, video_path, chunk_id
            )
            video_task = loop.run_in_executor(
                None, video_inference_analyzer.analyze, video_path, chunk_id
            )
            facial_task = loop.run_in_executor(
                None, facial_expression_analyzer.analyze, video_path, chunk_id
            )

            gaze_data = session.get_log_data()

            # Compute gaze summary stats from raw log entries
            gaze_counts = {"Looking Forward": 0, "Looking Left": 0,
                           "Looking Right": 0, "Looking Away": 0}
            for entry in gaze_data:
                s = entry.get("status", "")
                if s in gaze_counts:
                    gaze_counts[s] += 1
            gaze_total = sum(gaze_counts.values()) or 1  # avoid div-by-zero
            gaze_summary = {
                "total_frames": gaze_total,
                "looking_forward": gaze_counts["Looking Forward"],
                "looking_left":    gaze_counts["Looking Left"],
                "looking_right":   gaze_counts["Looking Right"],
                "looking_away":    gaze_counts["Looking Away"],
                "looking_forward_pct": round(gaze_counts["Looking Forward"] / gaze_total * 100, 1),
                "looking_left_pct":    round(gaze_counts["Looking Left"]    / gaze_total * 100, 1),
                "looking_right_pct":   round(gaze_counts["Looking Right"]   / gaze_total * 100, 1),
                "looking_away_pct":    round(gaze_counts["Looking Away"]    / gaze_total * 100, 1),
            } if gaze_data else {}

            voice_result, video_result, facial_result = await asyncio.gather(
                voice_task, video_task, facial_task
            )

            # Build a predictions list from the in-process VideoMAE result
            # so the frontend's existing ChunkResult shape stays compatible
            predictions = []
            if video_result.get("score") is not None:
                predictions = [{
                    "chunk": chunk_index,
                    "video_file": Path(video_path).name,
                    "confidence": video_result["score"],
                    "timestamp": time.time(),
                    "processing_time": 0,
                }]
            inference_summary = {
                "count": len(predictions),
                "mean_confidence": video_result.get("score") or 0.0,
                "min_confidence": video_result.get("score") or 0.0,
                "max_confidence": video_result.get("score") or 0.0,
            } if predictions else {}

            print(
                f"[Chunk {chunk_index}] Done: "
                f"gaze={len(gaze_data)} (L={gaze_summary.get('looking_left',0)} R={gaze_summary.get('looking_right',0)} F={gaze_summary.get('looking_forward',0)}), "
                f"confidence={video_result.get('score')}, "
                f"facial={facial_result.get('score')}, "
                f"voice={voice_result.get('score')}"
            )

            try:
                await websocket.send_json({
                    'type': 'chunk_processed',
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'gaze_data': gaze_data,
                    'gaze_summary': gaze_summary,
                    'predictions': predictions,
                    'inference_summary': inference_summary,
                    'voice_analysis': voice_result,
                    'facial_analysis': facial_result,
                })
            except Exception:
                print(f"[Chunk {chunk_index}] Client disconnected — results discarded")

        except Exception as e:
            print(f"[Chunk {chunk_index}] ERROR: {e}")
            try:
                await websocket.send_json({
                    'type': 'chunk_error',
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'message': str(e),
                })
            except Exception:
                print(f"[Chunk {chunk_index}] Could not send chunk_error — client disconnected")
        finally:
            session.stop()
            # ---- Clean up all files produced by this chunk ----
            files_to_delete = [
                video_path,
                str(session.log_file_path),
                str(session.vision_log),
                str(session.inference_log),
                str(session.predictions_file),
                str(Path(video_path).parent / f"voice_tmp_{chunk_id}.wav"),
            ]
            for fpath in files_to_delete:
                try:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                except Exception:
                    pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_session = None
    monitoring_task = None
    
    async def monitor_predictions():
        """Background task to monitor and send prediction updates"""
        while current_session and current_session.is_running:
            try:
                # Check for new predictions
                new_preds = current_session.get_new_predictions()
                
                if new_preds:
                    # Send each new prediction to frontend
                    for pred in new_preds:
                        await websocket.send_json({
                            'type': 'prediction_update',
                            'session_id': current_session.session_id,
                            'prediction': pred
                        })
                    
                    # Also send summary
                    all_data = current_session.get_predictions()
                    if all_data.get('summary'):
                        await websocket.send_json({
                            'type': 'prediction_summary',
                            'session_id': current_session.session_id,
                            'summary': all_data['summary']
                        })
                
                # Check every 2 seconds
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error in prediction monitoring: {e}")
                break
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            action = message.get('action')
            
            if action == 'start_session':
                # video_path is required — frontend must upload video first via POST /upload_video
                video_path = message.get('video_path', None)
                if not video_path:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'video_path is required. Upload the video via POST /upload_video first.'
                    })
                    continue

                session_id = str(uuid.uuid4())
                loop = message.get('loop', False)
                speed = float(message.get('speed', 1.0))
                session = VisionSession(session_id,
                                        video_path=video_path, loop=loop, speed=speed)
                # Run in executor — start() contains time.sleep() that would block the event loop
                await asyncio.get_event_loop().run_in_executor(None, session.start)

                active_sessions[session_id] = session
                current_session = session
                
                # Start monitoring predictions
                monitoring_task = asyncio.create_task(monitor_predictions())
                
                await websocket.send_json({
                    'type': 'session_started',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                })
                
            elif action == 'stop_session':
                if current_session:
                    print(f"Stopping session {current_session.session_id}")
                    current_session.stop()
                    
                    # Cancel monitoring task
                    if monitoring_task:
                        monitoring_task.cancel()
                        try:
                            await monitoring_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Wait longer for final logs and predictions to be written
                    await asyncio.sleep(2.0)
                    
                    # Read and send log data
                    log_data = current_session.get_log_data()
                    print(f"Read {len(log_data)} log entries from {current_session.log_file_path}")
                    
                    # Read final predictions
                    predictions_data = current_session.get_predictions()
                    print(f"Read {len(predictions_data.get('predictions', []))} predictions")
                    
                    if len(log_data) == 0:
                        print(f"WARNING: No log data found!")
                        print(f"   File exists: {current_session.log_file_path.exists()}")
                        if current_session.log_file_path.exists():
                            print(f"   File size: {current_session.log_file_path.stat().st_size} bytes")
                    
                    # Run voice analysis on the session video (non-blocking via thread)
                    voice_result = None
                    if current_session.video_path:
                        print(f"Running voice analysis on: {current_session.video_path}")
                        loop_ref = asyncio.get_event_loop()
                        voice_result = await loop_ref.run_in_executor(
                            None,
                            voice_analyzer.analyze,
                            current_session.video_path,
                            current_session.session_id
                        )
                        if voice_result.get("error"):
                            print(f"Voice analysis error: {voice_result['error']}")
                        else:
                            print(f"Voice score: {voice_result['score']}")

                    await websocket.send_json({
                        'type': 'session_ended',
                        'session_id': current_session.session_id,
                        'log_data': log_data,
                        'predictions': predictions_data.get('predictions', []),
                        'summary': predictions_data.get('summary', {}),
                        'voice_analysis': voice_result,
                        'start_time': current_session.start_time.isoformat(),
                        'end_time': datetime.now().isoformat()
                    })
                    
                    # Cleanup
                    active_sessions.pop(current_session.session_id, None)
                    current_session = None
                    
            elif action == 'process_chunk':
                # Each 15-s video chunk goes through vision.py, realtime_inference,
                # and voice_analyzer in parallel. Results come back as 'chunk_processed'.
                video_path = message.get('video_path')
                chunk_id = message.get('chunk_id', str(uuid.uuid4()))
                chunk_index = int(message.get('chunk_index', 0))

                if not video_path:
                    await websocket.send_json({
                        'type': 'chunk_error',
                        'chunk_id': chunk_id,
                        'message': 'video_path is required for process_chunk'
                    })
                elif not os.path.exists(video_path):
                    await websocket.send_json({
                        'type': 'chunk_error',
                        'chunk_id': chunk_id,
                        'message': f'Video file not found: {video_path}'
                    })
                else:
                    print(f"Processing chunk {chunk_index} ({chunk_id}): {video_path}")
                    # Fire and forget — each chunk runs independently
                    asyncio.create_task(
                        _process_chunk(websocket, chunk_id, chunk_index, video_path)
                    )

            elif action == 'get_status':
                if current_session and current_session.is_running:
                    await websocket.send_json({
                        'type': 'status',
                        'session_id': current_session.session_id,
                        'is_running': True
                    })
                else:
                    await websocket.send_json({
                        'type': 'status',
                        'is_running': False
                    })
                    
    except WebSocketDisconnect:
        print("Client disconnected")
        if current_session:
            current_session.cleanup()
            active_sessions.pop(current_session.session_id, None)
        if monitoring_task:
            monitoring_task.cancel()
    except Exception as e:
        print(f"Error: {e}")
        if current_session:
            current_session.cleanup()
            active_sessions.pop(current_session.session_id, None)
        if monitoring_task:
            monitoring_task.cancel()


@app.post("/analyze_voice")
async def analyze_voice(file: UploadFile = File(...)):
    """Upload an audio or video file and receive a voice analysis report.

    Returns:
        score         - overall speaking skills score (0–1)
        window_times  - time axis for sliding window (seconds)
        window_scores - per-window scores
        energy        - loudness proxy array
        pitch_proxy   - zero-crossing-rate array (pitch proxy)
    """
    upload_dir = Path(__file__).parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / file_id

    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    print(f"[analyze_voice] Received: {file_path}")

    # Run analysis in thread pool so we don't block the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, voice_analyzer.analyze, str(file_path), file_id
    )

    return {"status": "ok", **result}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file from the frontend and return its server-local path.

    Frontend flow:
      1. POST /upload_video  →  receive video_path
      2. WS start_session(video_path=...)
    """
    upload_dir = Path(__file__).parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / file_id

    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # stream in 1 MB chunks
            f.write(chunk)

    print(f"Uploaded video: {file_path}")
    return {
        "status": "uploaded",
        "video_path": str(file_path)
    }


@app.get("/")
async def root():
    return {
        "message": "Vision Gaze Tracking Server",
        "status": "running",
        "active_sessions": len(active_sessions)
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    print("=" * 60)
    print("Vision Gaze Tracking WebSocket Server")
    print("=" * 60)
    print("Server starting on: http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
