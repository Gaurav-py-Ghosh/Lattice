"""
Real-time VideoMAE Confidence Inference for Interview Recordings

Monitors Vision/data/ directory for new video chunks and runs VideoMAE inference
in parallel with recording. Outputs confidence predictions for streaming to frontend.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import json
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast

# Add Atempt2 to path to import models
atempt2_path = Path(__file__).parent.parent / "Atempt2"
sys.path.insert(0, str(atempt2_path))

from src.model.video_model import VideoModel
from transformers import AutoImageProcessor


class RealtimeVideoInference:
    """Manages real-time VideoMAE inference on recorded chunks"""
    
    def __init__(self, session_id: str, model_path: str, output_dir: Path):
        self.session_id = session_id
        self.model_path = model_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Predictions storage
        self.predictions = []
        self.processed_chunks = set()
        
        # Output files
        self.predictions_file = output_dir / f"{session_id}_predictions.json"
        
        # Load model and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading VideoMAE model from: {model_path}", flush=True)
        print(f"   Device: {self.device}", flush=True)
        
        # Initialize processor
        self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        
        # Load model (VideoModel doesn't need num_outputs, it outputs 1 scalar by default)
        self.model = VideoModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Model loaded successfully", flush=True)
        
    def extract_frames(self, video_path: Path, num_frames: int = 16) -> np.ndarray:
        """Extract uniformly sampled frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Sample frame indices uniformly
        if total_frames < num_frames:
            # Repeat frames if video is too short
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) != num_frames:
            raise ValueError(f"Expected {num_frames} frames, got {len(frames)}")
        
        return np.array(frames)
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess frames for VideoMAE model"""
        # frames shape: (num_frames, H, W, C)
        
        # Convert to list of individual frames for the processor
        # Processor expects list of (H, W, C) arrays
        frames_list = [frame for frame in frames]
        
        # Use processor to normalize and resize
        # Returns pixel_values in shape (1, C, T, H, W) - already correct format!
        inputs = self.processor(
            frames_list,
            return_tensors="pt"
        )
        
        pixel_values = inputs['pixel_values']  # Shape: (1, C, T, H, W)
        
        return pixel_values.to(self.device)
    
    def predict_confidence(self, video_path: Path) -> float:
        """Run inference on a video chunk"""
        with torch.no_grad():
            # Extract frames
            frames = self.extract_frames(video_path, num_frames=16)
            
            # Preprocess
            video_tensor = self.preprocess_frames(frames)
            
            # Run inference with mixed precision
            with autocast(enabled=self.device.type == 'cuda'):
                output = self.model(video_tensor)
            
            # Extract scalar confidence score
            confidence = output.item()
            
            return confidence
    
    def _wait_for_video_ready(self, video_path: Path, max_attempts: int = 10, wait_time: float = 1.0) -> bool:
        """Wait for video file to be fully written and readable"""
        for attempt in range(max_attempts):
            try:
                # Try to open the video file
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # If we can read frame count and it's > 0, file is ready
                if frame_count > 0:
                    return True
                    
            except Exception as e:
                pass
            
            # Wait before retrying
            if attempt < max_attempts - 1:
                time.sleep(wait_time)
        
        return False
    
    def save_predictions(self):
        """Save predictions to JSON file"""
        data = {
            'session_id': self.session_id,
            'predictions': self.predictions,
            'summary': {
                'count': len(self.predictions),
                'mean_confidence': float(np.mean([p['confidence'] for p in self.predictions])) if self.predictions else 0.0,
                'min_confidence': float(np.min([p['confidence'] for p in self.predictions])) if self.predictions else 0.0,
                'max_confidence': float(np.max([p['confidence'] for p in self.predictions])) if self.predictions else 0.0,
            }
        }
        
        with open(self.predictions_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved predictions: {self.predictions_file}", flush=True)
    
    def process_chunk(self, video_path: Path) -> dict:
        """Process a single video chunk"""
        chunk_num = int(video_path.stem.split('_chunk')[1].split('_')[0])
        
        print(f"\nProcessing chunk {chunk_num}: {video_path.name}", flush=True)
        start_time = time.time()
        
        try:
            confidence = self.predict_confidence(video_path)
            elapsed = time.time() - start_time
            
            prediction = {
                'chunk': chunk_num,
                'video_file': video_path.name,
                'confidence': float(confidence),
                'timestamp': time.time(),
                'processing_time': elapsed
            }
            
            self.predictions.append(prediction)
            self.processed_chunks.add(chunk_num)
            
            print(f"   Confidence: {confidence:.4f} (processed in {elapsed:.2f}s)", flush=True)
            
            # Save after each prediction for real-time updates
            self.save_predictions()
            
            return prediction
            
        except Exception as e:
            print(f"ERROR: Error processing chunk {chunk_num}: {e}", flush=True)
            return None
    
    def monitor_directory(self, data_dir: Path, check_interval: float = 2.0):
        """Monitor directory for new video chunks and process them"""
        print(f"\nMonitoring: {data_dir}", flush=True)
        print(f"   Session ID: {self.session_id}", flush=True)
        print(f"   Check interval: {check_interval}s", flush=True)
        print(f"   Output: {self.predictions_file}", flush=True)
        print("\nWaiting for video chunks...\n", flush=True)
        
        # Check for stop signal file
        stop_signal_file = data_dir / f"{self.session_id}_stop.signal"
        
        while True:
            # Find all video chunks for this session
            video_pattern = f"{self.session_id}_chunk*.mp4"
            video_files = sorted(data_dir.glob(video_pattern))
            
            # Process new chunks
            for video_path in video_files:
                chunk_num = int(video_path.stem.split('_chunk')[1].split('_')[0])
                
                if chunk_num not in self.processed_chunks:
                    # Wait for file to be properly closed and readable
                    if self._wait_for_video_ready(video_path):
                        self.process_chunk(video_path)
                    else:
                        print(f"   Skipping chunk {chunk_num} - file not ready", flush=True)
            
            # Check for stop signal
            if stop_signal_file.exists():
                print(f"\nStop signal detected", flush=True)
                
                # Process any remaining chunks one more time
                video_files = sorted(data_dir.glob(video_pattern))
                for video_path in video_files:
                    chunk_num = int(video_path.stem.split('_chunk')[1].split('_')[0])
                    if chunk_num not in self.processed_chunks:
                        if self._wait_for_video_ready(video_path):
                            self.process_chunk(video_path)
                
                # Clean up stop signal
                stop_signal_file.unlink()
                
                print(f"\nInference complete:")
                print(f"   Total chunks processed: {len(self.predictions)}")
                if self.predictions:
                    print(f"   Mean confidence: {np.mean([p['confidence'] for p in self.predictions]):.4f}")
                
                break
            
            # Wait before next check
            time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description='Real-time VideoMAE Inference')
    parser.add_argument('--session-id', type=str, required=True, help='Session ID to monitor')
    parser.add_argument('--model-path', type=str, 
                        default=str(Path(__file__).parent.parent / "Atempt2" / "checkpoints" / "videoMAE_confidence_ranker_epoch6.pth"),
                        help='Path to VideoMAE model checkpoint')
    parser.add_argument('--data-dir', type=str,
                        default=str(Path(__file__).parent / "data"),
                        help='Directory containing video chunks')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / "data" / "predictions"),
                        help='Directory for prediction outputs')
    parser.add_argument('--check-interval', type=float, default=2.0,
                        help='Seconds between directory checks')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Real-time VideoMAE Confidence Inference")
    print("=" * 70)
    
    # Verify model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    # Create inference engine
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    try:
        engine = RealtimeVideoInference(
            session_id=args.session_id,
            model_path=str(model_path),
            output_dir=output_dir
        )
        
        # Start monitoring
        engine.monitor_directory(data_dir, check_interval=args.check_interval)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
