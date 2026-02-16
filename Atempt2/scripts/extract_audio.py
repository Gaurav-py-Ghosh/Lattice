import os
import json
import pandas as pd
import subprocess
from tqdm import tqdm

# --- Configuration ---
# These paths should match those in your training scripts
METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata.jsonl"
VIDEO_ROOT = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\videos"
AUDIO_OUTPUT_ROOT = os.path.join(os.path.dirname(VIDEO_ROOT), "audio") # New directory for audio

def extract_audio_from_videos():
    """
    Extracts audio from video files using a direct ffmpeg command via subprocess.
    This is more robust than using moviepy if there are environment issues.
    """
    print(f"Loading metadata from: {METADATA_PATH}")
    data = []
    with open(METADATA_PATH, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    metadata = pd.DataFrame(data)

    os.makedirs(AUDIO_OUTPUT_ROOT, exist_ok=True)
    print(f"Audio will be saved to: {AUDIO_OUTPUT_ROOT}")

    new_metadata_records = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting Audio"):
        relative_video_path = row['file_name']
        if relative_video_path.startswith('videos/'):
            relative_video_path = relative_video_path[len('videos/'):]
        video_path = os.path.join(VIDEO_ROOT, relative_video_path)

        # Create a corresponding path for the audio file
        # Ensure the subdirectory structure is maintained
        audio_filename = os.path.splitext(relative_video_path)[0] + ".wav"
        audio_path = os.path.join(AUDIO_OUTPUT_ROOT, audio_filename)
        
        # Ensure the output directory for the specific audio file exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}, skipping.")
            continue
        
        if os.path.exists(audio_path):
            pass # Skip if audio already exists
        else:
            try:
                # Construct the ffmpeg command
                # -i: input file
                # -vn: no video output
                # -acodec pcm_s16le: standard WAV codec
                # -ar 16000: sample rate 16kHz (common for speech models)
                # -ac 1: mono channel
                # -y: overwrite output file if it exists
                # -hide_banner -loglevel error: suppress verbose output
                command = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vn',
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    '-hide_banner',
                    '-loglevel', 'error',
                    audio_path
                ]
                
                # Execute the command
                subprocess.run(command, check=True)

            except subprocess.CalledProcessError as e:
                print(f"ffmpeg error extracting audio from {video_path}: {e}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred with {video_path}: {e}")
                continue
        
        # Add the audio path to the new metadata record
        row['audio_path'] = audio_path
        new_metadata_records.append(row.to_dict())

    # Save the new metadata with audio paths
    new_metadata_path = os.path.join(os.path.dirname(METADATA_PATH), "metadata_with_audio.jsonl")
    with open(new_metadata_path, 'w') as f:
        for record in new_metadata_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nAudio extraction complete. New metadata saved to: {new_metadata_path}")
    print(f"Extracted audio files are in: {AUDIO_OUTPUT_ROOT}")

if __name__ == '__main__':
    extract_audio_from_videos()
