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

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
# FastAPI lifespan — eager-load the voice model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Load the voice model in a thread-pool worker at startup so the event
    loop isn't blocked while torch initialises transformers weights."""
    loop = asyncio.get_event_loop()
    print("[Startup] Pre-loading VoiceAnalyzer model…")
    await loop.run_in_executor(None, voice_analyzer.load)
    print("[Startup] VoiceAnalyzer ready.")
    yield  # server runs here
    # (cleanup on shutdown goes after yield if needed)

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
            vision_log_file = open(self.vision_log, 'w', encoding='utf-8')
            # PYTHONIOENCODING ensures emoji/unicode in vision.py print() go to the log cleanly
            child_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            if os.name == 'nt':  # Windows
                self.vision_process = subprocess.Popen(
                    vision_cmd,
                    cwd=Path(__file__).parent,
                    stdout=vision_log_file,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Linux/Mac
                self.vision_process = subprocess.Popen(
                    vision_cmd,
                    cwd=Path(__file__).parent,
                    stdout=vision_log_file,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )
            print(f"Started vision.py (PID: {self.vision_process.pid})")
            print(f"   Output log: {self.vision_log}")
        except Exception as e:
            print(f"ERROR: starting vision.py: {e}")
            self.is_running = False
            return
        
        # Give vision.py a moment to start before launching inference
        time.sleep(2)
        
        # Start realtime_inference.py with output logging
        try:
            inference_log_file = open(self.inference_log, 'w', encoding='utf-8')
            if os.name == 'nt':  # Windows
                self.inference_process = subprocess.Popen(
                    inference_cmd,
                    cwd=Path(__file__).parent,
                    stdout=inference_log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Linux/Mac
                self.inference_process = subprocess.Popen(
                    inference_cmd,
                    cwd=Path(__file__).parent,
                    stdout=inference_log_file,
                    stderr=subprocess.STDOUT,
                )
            print(f"Started realtime_inference.py (PID: {self.inference_process.pid})")
            print(f"   Output log: {self.inference_log}")
        except Exception as e:
            print(f"ERROR: starting inference: {e}")
            # Continue anyway, vision still works without inference
        
        self.is_running = True
        
        # Check if processes are still running
        time.sleep(0.5)
        if self.vision_process.poll() is not None:
            print(f"ERROR: vision.py crashed immediately (exit code: {self.vision_process.returncode})")
            self.is_running = False
            return
        
        print(f"Session started: {self.session_id}")
        print(f"   Log file: {self.log_file_path}")
        print(f"   Predictions file: {self.predictions_file}")
        
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

    1. vision.py          — gaze tracking (subprocess, exits at EOF)
    2. realtime_inference — VideoMAE confidence (subprocess, coupled to vision.py)
    3. voice_analyzer     — speaking skills (in-process, runs concurrently)

    Results are sent back as a 'chunk_processed' WebSocket message.
    """
    loop = asyncio.get_event_loop()

    # ---- Create a mini VisionSession for this chunk ----
    session = VisionSession(chunk_id, video_path=video_path)
    session.start()

    if not session.is_running:
        try:
            await websocket.send_json({
                'type': 'chunk_error',
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'message': f'Failed to start vision session for chunk {chunk_id}',
            })
        except Exception:
            pass
        return

    try:
        # ---- Wait for vision.py to finish processing the video (EOF) ----
        # Run in executor so we don't block the event loop
        await loop.run_in_executor(None, session.vision_process.wait)
        print(f"[Chunk {chunk_index}] vision.py finished (PID {session.vision_process.pid})")

        # Give realtime_inference a moment to process the chunks vision.py wrote
        await asyncio.sleep(3)

        # ---- Run voice analysis concurrently while inference settles ----
        voice_task = loop.run_in_executor(
            None, voice_analyzer.analyze, video_path, chunk_id
        )

        # ---- Stop inference process (vision.py already done, give it 3 s then kill) ----
        def _stop_inference():
            proc = session.inference_process
            if proc and proc.poll() is None:  # still running
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        await loop.run_in_executor(None, _stop_inference)

        # ---- Collect results ----
        gaze_data = session.get_log_data()
        pred_data = session.get_predictions()
        voice_result = await voice_task

        print(
            f"[Chunk {chunk_index}] Done: "
            f"gaze={len(gaze_data)}, "
            f"preds={len(pred_data.get('predictions', []))}, "
            f"voice={voice_result.get('score')}"
        )

        try:
            await websocket.send_json({
                'type': 'chunk_processed',
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'gaze_data': gaze_data,
                'predictions': pred_data.get('predictions', []),
                'inference_summary': pred_data.get('summary', {}),
                'voice_analysis': voice_result,
            })
        except Exception:
            print(f"[Chunk {chunk_index}] Client already disconnected — results discarded")

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
                session.start()
                
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
