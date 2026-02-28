"""
Validation Test: Compare model predictions with ground truth confidence scores
"""
import torch
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from transformers import AutoImageProcessor

# Add project paths
sys.path.append(str(Path(__file__).parent))
from src.model.video_model import VideoModel

class ConfidenceValidator:
    def __init__(self, model_path, metadata_path, video_root):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load metadata
        self.metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        print(f"Loaded {len(self.metadata)} video metadata entries")
        
        self.video_root = Path(video_root)
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = VideoModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("Model loaded successfully")
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        print("Processor loaded")
    
    def extract_frames(self, video_path, num_frames=16):
        """Extract uniformly sampled frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Sample frame indices uniformly
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
        
        return frames
    
    def preprocess_frames(self, frames):
        """Preprocess frames for VideoMAE model"""
        # Process frames - returns (1, C, T, H, W)
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        return pixel_values
    
    def predict_confidence(self, video_path):
        """Run inference on a video"""
        with torch.no_grad():
            frames = self.extract_frames(video_path, num_frames=16)
            video_tensor = self.preprocess_frames(frames)
            
            # Run inference
            output = self.model(video_tensor)
            confidence = output.item()
            
            return confidence
    
    def validate_video(self, video_id):
        """Validate a single video"""
        # Find metadata entry
        entry = None
        for item in self.metadata:
            if item['id'] == video_id:
                entry = item
                break
        
        if entry is None:
            print(f"ERROR: Video {video_id} not found in metadata")
            return None
        
        # Get video path
        video_filename = entry['file_name']
        if video_filename.startswith('videos/'):
            video_filename = video_filename[len('videos/'):]
        
        video_path = self.video_root / video_filename
        
        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}")
            return None
        
        # Get ground truth
        ground_truth = entry['confidence_score']
        
        # Predict
        print(f"\nProcessing: {video_path.name}")
        print(f"  Duration: {entry['duration']}")
        print(f"  Question: {entry['question']}")
        try:
            predicted = self.predict_confidence(video_path)
            
            # Calculate error
            error = predicted - ground_truth
            abs_error = abs(error)
            percent_error = (abs_error / (abs(ground_truth) + 1e-6)) * 100
            
            result = {
                'video_id': video_id,
                'video_file': video_path.name,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'error': error,
                'abs_error': abs_error,
                'percent_error': percent_error
            }
            
            return result
            
        except Exception as e:
            print(f"ERROR: Failed to process video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_validation(self, video_ids):
        """Run validation on multiple videos"""
        print("\n" + "="*70)
        print("VALIDATION TEST: Model Predictions vs Ground Truth")
        print("="*70)
        
        results = []
        for vid_id in video_ids:
            result = self.validate_video(vid_id)
            if result:
                results.append(result)
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        for r in results:
            print(f"\nVideo: {r['video_file']}")
            print(f"  Ground Truth:  {r['ground_truth']:+.6f}")
            print(f"  Predicted:     {r['predicted']:+.6f}")
            print(f"  Error:         {r['error']:+.6f}")
            print(f"  Abs Error:     {r['abs_error']:.6f}")
            print(f"  Percent Error: {r['percent_error']:.2f}%")
        
        if len(results) >= 2:
            print("\n" + "="*70)
            print("VALIDATION METRICS")
            print("="*70)
            
            errors = [r['abs_error'] for r in results]
            mean_abs_error = np.mean(errors)
            
            gt_scores = [r['ground_truth'] for r in results]
            pred_scores = [r['predicted'] for r in results]
            
            # Check if relative ordering is preserved
            gt_diffs = []
            pred_diffs = []
            for i in range(len(results)-1):
                for j in range(i+1, len(results)):
                    gt_diff = gt_scores[j] - gt_scores[i]
                    pred_diff = pred_scores[j] - pred_scores[i]
                    gt_diffs.append(gt_diff)
                    pred_diffs.append(pred_diff)
            
            # Calculate correlation
            from scipy.stats import spearmanr
            if len(gt_scores) > 1:
                corr, p_value = spearmanr(gt_scores, pred_scores)
                print(f"Spearman Correlation: {corr:.4f} (p={p_value:.4f})")
            
            print(f"Mean Absolute Error: {mean_abs_error:.6f}")
            
            # Check ranking preservation
            ranking_preserved = 0
            total_pairs = 0
            for i in range(len(gt_diffs)):
                if (gt_diffs[i] > 0 and pred_diffs[i] > 0) or (gt_diffs[i] < 0 and pred_diffs[i] < 0):
                    ranking_preserved += 1
                total_pairs += 1
            
            if total_pairs > 0:
                ranking_accuracy = (ranking_preserved / total_pairs) * 100
                print(f"Ranking Preservation: {ranking_preserved}/{total_pairs} ({ranking_accuracy:.1f}%)")
        
        return results


if __name__ == "__main__":
    # Paths
    MODEL_PATH = r"C:\Users\gaura\PRJ\Atempt2\checkpoints\videoMAE_confidence_ranker_epoch6.pth"
    METADATA_PATH = r"C:\Users\gaura\.cache\huggingface\hub\datasets--AI4A-lab--RecruitView\snapshots\0cfa07ed0a43622f9104592b100d7bf3a25f6140\metadata.jsonl"
    VIDEO_ROOT = r"C:\Users\gaura\.cache\huggingface\hub\datasets--AI4A-lab--RecruitView\snapshots\0cfa07ed0a43622f9104592b100d7bf3a25f6140\videos"
    
    # Test videos with different confidence scores
    # From metadata:
    # 0001: confidence = -0.362 (low confidence)
    # 0003: confidence = 0.600 (good confidence)  
    # 0004: confidence = 1.289 (very good confidence)
    # 2011: confidence = -1.832 (very low confidence)
    
    test_videos = ['0001', '0003', '0004']
    
    # Run validation
    validator = ConfidenceValidator(MODEL_PATH, METADATA_PATH, VIDEO_ROOT)
    results = validator.run_validation(test_videos)
    
    print("\n" + "="*70)
    print("Validation test complete!")
    print("="*70)
