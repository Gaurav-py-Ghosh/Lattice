/**
 * useChunkedVisionSession Hook
 *
 * Replaces the old WebSocket-based session approach.
 * - Shows live webcam via a video ref
 * - Records in 15-second chunks using MediaRecorder
 * - POSTs each chunk to the FastAPI server: /session/{id}/chunk
 * - Returns per-chunk gaze entries and predictions in real time
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const SERVER_URL = 'http://localhost:8000';
const CHUNK_DURATION_MS = 15_000; // 15 seconds

export interface GazeEntry {
  timestamp: string;
  status: string;
}

export interface PredictionEntry {
  chunk: number;
  video_file: string;
  confidence: number;
  timestamp: number;
  processing_time: number;
}

export interface ChunkResult {
  chunk: number;
  gaze_entries: GazeEntry[];
  predictions: PredictionEntry[];
  summary: Record<string, number>;
}

export interface SessionSummary {
  session_id: string;
  start_time: string;
  end_time: string;
  chunks_processed: number;
  gaze_entries: GazeEntry[];
  predictions: PredictionEntry[];
  summary: Record<string, number>;
}

export function useChunkedVisionSession(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chunkResults, setChunkResults] = useState<ChunkResult[]>([]);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [latestConfidence, setLatestConfidence] = useState<number | null>(null);
  const [gazeEntries, setGazeEntries] = useState<GazeEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isProcessingChunk, setIsProcessingChunk] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionIdRef = useRef<string | null>(null); // stable ref for callbacks
  const chunksQueue = useRef<Blob[]>([]);
  const isStoppingRef = useRef(false);

  // ── Attach webcam stream to video element ────────────────────────────────
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      return stream;
    } catch (err) {
      setError(`Webcam access denied: ${err}`);
      throw err;
    }
  }, [videoRef]);

  const stopWebcam = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [videoRef]);

  // ── Upload a single chunk blob to the server ─────────────────────────────
  const uploadChunk = useCallback(async (blob: Blob, sidOverride?: string) => {
    const sid = sidOverride ?? sessionIdRef.current;
    if (!sid) return;

    setIsProcessingChunk(true);
    try {
      const form = new FormData();
      form.append('video', blob, `chunk.webm`);

      const res = await fetch(`${SERVER_URL}/session/${sid}/chunk`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        console.error('[chunk] server error', res.status, await res.text());
        return;
      }

      const result: ChunkResult = await res.json();
      console.log(`[chunk ${result.chunk}] gaze=${result.gaze_entries.length} preds=${result.predictions.length}`);

      setChunkResults(prev => [...prev, result]);
      setGazeEntries(prev => [...prev, ...result.gaze_entries]);

      if (result.predictions.length > 0) {
        setLatestConfidence(result.predictions.at(-1)!.confidence);
      }
    } catch (err) {
      console.error('[chunk] upload error', err);
    } finally {
      setIsProcessingChunk(false);
    }
  }, []);

  // ── Start a new 15-second recording segment ───────────────────────────────
  const startNextChunk = useCallback((stream: MediaStream) => {
    if (isStoppingRef.current) return;

    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : MediaRecorder.isTypeSupported('video/webm')
      ? 'video/webm'
      : 'video/mp4';

    const recorder = new MediaRecorder(stream, { mimeType });
    const blobs: BlobPart[] = [];

    recorder.ondataavailable = e => {
      if (e.data.size > 0) blobs.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(blobs, { type: mimeType });
      uploadChunk(blob);
    };

    recorder.start();
    mediaRecorderRef.current = recorder;

    // Schedule stop + next chunk after CHUNK_DURATION_MS
    chunkTimerRef.current = setTimeout(() => {
      if (!isStoppingRef.current) {
        recorder.stop();
        startNextChunk(stream);
      } else {
        recorder.stop();
      }
    }, CHUNK_DURATION_MS);
  }, [uploadChunk]);

  // ── Public: start session ─────────────────────────────────────────────────
  const startSession = useCallback(async () => {
    setError(null);
    setChunkResults([]);
    setSessionSummary(null);
    setGazeEntries([]);
    setLatestConfidence(null);
    isStoppingRef.current = false;

    // 1. Start webcam
    let stream: MediaStream;
    try {
      stream = await startWebcam();
    } catch {
      return;
    }

    // 2. Create server session
    try {
      const res = await fetch(`${SERVER_URL}/session/start`, { method: 'POST' });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const { session_id } = await res.json();
      setSessionId(session_id);
      sessionIdRef.current = session_id;
      console.log('[session] created', session_id);
    } catch (err) {
      setError(`Could not start server session: ${err}`);
      stopWebcam();
      return;
    }

    // 3. Begin chunked recording
    setIsSessionActive(true);
    startNextChunk(stream);
  }, [startWebcam, stopWebcam, startNextChunk]);

  // ── Public: stop session ──────────────────────────────────────────────────
  const stopSession = useCallback(async () => {
    isStoppingRef.current = true;

    // Stop the timer so no new chunk is started
    if (chunkTimerRef.current) {
      clearTimeout(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }

    // Stop current recorder (will trigger onstop → uploadChunk for remaining data)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    stopWebcam();
    setIsSessionActive(false);

    const sid = sessionIdRef.current;
    if (!sid) return;

    // Small delay to let the last chunk upload complete
    await new Promise(r => setTimeout(r, 2000));

    // Finalise session on server
    try {
      const res = await fetch(`${SERVER_URL}/session/${sid}/stop`, { method: 'POST' });
      if (res.ok) {
        const summary: SessionSummary = await res.json();
        setSessionSummary(summary);
        console.log('[session] stopped', summary);
      }
    } catch (err) {
      console.error('[session] stop error', err);
    }

    setSessionId(null);
    sessionIdRef.current = null;
  }, [stopWebcam]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isStoppingRef.current = true;
      if (chunkTimerRef.current) clearTimeout(chunkTimerRef.current);
      mediaRecorderRef.current?.stop();
      stopWebcam();
    };
  }, [stopWebcam]);

  return {
    // State
    isSessionActive,
    sessionId,
    chunkResults,
    sessionSummary,
    gazeEntries,
    latestConfidence,
    isProcessingChunk,
    error,
    // Actions
    startSession,
    stopSession,
  };
}
