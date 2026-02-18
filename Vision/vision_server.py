"""
Vision Gaze Tracking WebSocket Server

This server manages vision.py sessions and communicates with the Next.js frontend.
- Starts/stops vision.py subprocess
- Starts/stops realtime_inference.py for VideoMAE confidence prediction
- Manages session-specific log files
- Sends gaze data and confidence predictions to frontend via WebSocket
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_sessions = {}


class VisionSession:
    def __init__(self, session_id: str, headless: bool = True):
        self.session_id = session_id
        self.headless = headless  # Default to headless (no windows)
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
        """Start vision.py and realtime_inference.py subprocesses with session ID"""
        # Ensure data directory exists
        self.log_file_path.parent.mkdir(exist_ok=True)
        self.predictions_file.parent.mkdir(exist_ok=True)
        
        # Use sys.executable to ensure we use the same Python (from conda env)
        python_exe = sys.executable
        vision_script = Path(__file__).parent / "vision.py"
        inference_script = Path(__file__).parent / "realtime_inference.py"
        
        print(f"Starting vision system with:")
        print(f"   Python: {python_exe}")
        print(f"   Vision Script: {vision_script}")
        print(f"   Inference Script: {inference_script}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Headless Mode: {self.headless}")
        
        # Build command arguments for vision.py
        vision_cmd = [python_exe, str(vision_script), "--session-id", self.session_id]
        if self.headless:
            vision_cmd.append("--headless")
        
        # Build command arguments for inference.py
        inference_cmd = [
            python_exe, str(inference_script),
            "--session-id", self.session_id
        ]
        
        # Start vision.py with output logging
        try:
            vision_log_file = open(self.vision_log, 'w', encoding='utf-8')
            if os.name == 'nt':  # Windows
                self.vision_process = subprocess.Popen(
                    vision_cmd,
                    cwd=Path(__file__).parent,
                    stdout=vision_log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Linux/Mac
                self.vision_process = subprocess.Popen(
                    vision_cmd,
                    cwd=Path(__file__).parent,
                    stdout=vision_log_file,
                    stderr=subprocess.STDOUT,
                )
            print(f"Started vision.py (PID: {self.vision_process.pid})")
            print(f"   Output log: {self.vision_log}")
        except Exception as e:
            print(f"ERROR: starting vision.py: {e}")
            self.is_running = False
            return
        
        # Give vision.py a moment to start and create first chunk
        import time
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
                # Create new session
                session_id = str(uuid.uuid4())
                # headless=True means no OpenCV windows, headless=False shows windows (minimized)
                headless = message.get('headless', True)  # Default to headless
                session = VisionSession(session_id, headless=headless)
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
                    
                    await websocket.send_json({
                        'type': 'session_ended',
                        'session_id': current_session.session_id,
                        'log_data': log_data,
                        'predictions': predictions_data.get('predictions', []),
                        'summary': predictions_data.get('summary', {}),
                        'start_time': current_session.start_time.isoformat(),
                        'end_time': datetime.now().isoformat()
                    })
                    
                    # Cleanup
                    active_sessions.pop(current_session.session_id, None)
                    current_session = None
                    
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
