import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
import collections

# --- Configuration ---
MODEL_SIZE = "Oriserve/Whisper-Hindi2Hinglish-Swift"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCK_DURATION_MS = 100
TRANSCRIPTION_INTERVAL_S = 10

# --- Global State ---
audio_buffer = collections.deque()

def audio_callback(indata, frames, time, status):
    """This function is called by the sounddevice stream for each new audio chunk."""
    if status:
        print(f"Audio Status: {status}")
    audio_buffer.append(indata.copy())

def main():
    """Main function to run the transcription loop."""
    print(f"Loading faster-whisper model '{MODEL_SIZE}'... (This may take a moment on first run)")
    try:
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please ensure you have an internet connection for the first run, and that the model name is correct.")
        return

    print(f"Starting audio stream... Transcribing every {TRANSCRIPTION_INTERVAL_S} seconds.")
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype=DTYPE, 
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
        ):
            while True:
                time.sleep(TRANSCRIPTION_INTERVAL_S)
                
                if not audio_buffer:
                    continue

                # Concatenate all audio chunks from the buffer
                # The deque might have been cleared by a previous iteration, so check again
                try:
                    all_chunks = np.concatenate(list(audio_buffer))
                    audio_buffer.clear()
                except ValueError:
                    continue # Buffer was cleared between check and concatenation
                
                # Convert from int16 to float32 and normalize
                audio_float32 = all_chunks.flatten().astype(np.float32) / 32768.0
                
                print(f"Transcribing {len(audio_float32)/SAMPLE_RATE:.2f}s of audio...")
                
                # Transcribe the numpy array directly in memory
                segments, info = model.transcribe(audio_float32, beam_size=5)
                
                transcription = "".join([s.text for s in segments]).strip()
                
                if transcription:
                    print(f"Transcription: {transcription}\n")
                else:
                    print("No speech detected.\n")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()