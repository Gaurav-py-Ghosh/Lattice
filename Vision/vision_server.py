"""
Vision Gaze Tracking WebSocket Server

This server manages vision.py sessions and communicates with the Next.js frontend.
- Starts/stops vision.py subprocess
- Manages session-specific log files
- Sends gaze data to frontend via WebSocket
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
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.process: Optional[subprocess.Popen] = None
        self.log_file_path = Path(__file__).parent / "data" / f"gaze_log_{session_id}.txt"
        self.start_time = datetime.now()
        self.is_running = False
        
    def start(self):
        """Start vision.py subprocess with session ID"""
        # Ensure data directory exists
        self.log_file_path.parent.mkdir(exist_ok=True)
        
        # Use sys.executable to ensure we use the same Python (from conda env)
        python_exe = sys.executable
        vision_script = Path(__file__).parent / "vision.py"
        
        print(f"📍 Starting vision.py with:")
        print(f"   Python: {python_exe}")
        print(f"   Script: {vision_script}")
        print(f"   Session ID: {self.session_id}")
        
        # Start vision.py with session ID as argument
        # CREATE_NEW_CONSOLE opens a new window for vision.py
        try:
            if os.name == 'nt':  # Windows
                self.process = subprocess.Popen(
                    [python_exe, str(vision_script), "--session-id", self.session_id],
                    cwd=Path(__file__).parent,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:  # Linux/Mac
                self.process = subprocess.Popen(
                    [python_exe, str(vision_script), "--session-id", self.session_id],
                    cwd=Path(__file__).parent,
                )
        except Exception as e:
            print(f"❌ ERROR starting vision.py: {e}")
            self.is_running = False
            return
            
        self.is_running = True
        
        # Give it a moment to start
        import time
        time.sleep(0.5)
        
        # Check if it's still running
        if self.process.poll() is not None:
            print(f"❌ ERROR: vision.py crashed immediately (exit code: {self.process.returncode})")
            self.is_running = False
            return
        
        print(f"✓ Started vision.py session: {self.session_id} (PID: {self.process.pid})")
        print(f"   Log file: {self.log_file_path}")
        
    def stop(self):
        """Stop vision.py subprocess gracefully"""
        if self.process:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if doesn't terminate
                self.process.kill()
                self.process.wait()
            
            print(f"✓ Stopped vision.py session: {self.session_id}")
            self.is_running = False
            
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
    
    def cleanup(self):
        """Clean up session resources"""
        self.stop()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_session = None
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            action = message.get('action')
            
            if action == 'start_session':
                # Create new session
                session_id = str(uuid.uuid4())
                session = VisionSession(session_id)
                session.start()
                
                active_sessions[session_id] = session
                current_session = session
                
                await websocket.send_json({
                    'type': 'session_started',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                })
                
            elif action == 'stop_session':
                if current_session:
                    print(f"⏹️  Stopping session {current_session.session_id}")
                    current_session.stop()
                    
                    # Wait longer for final logs to be written and file to close
                    await asyncio.sleep(1.5)
                    
                    # Read and send log data
                    log_data = current_session.get_log_data()
                    print(f"📊 Read {len(log_data)} log entries from {current_session.log_file_path}")
                    
                    if len(log_data) == 0:
                        print(f"⚠️  WARNING: No log data found!")
                        print(f"   File exists: {current_session.log_file_path.exists()}")
                        if current_session.log_file_path.exists():
                            print(f"   File size: {current_session.log_file_path.stat().st_size} bytes")
                    
                    await websocket.send_json({
                        'type': 'session_ended',
                        'session_id': current_session.session_id,
                        'log_data': log_data,
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
    except Exception as e:
        print(f"Error: {e}")
        if current_session:
            current_session.cleanup()
            active_sessions.pop(current_session.session_id, None)


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
