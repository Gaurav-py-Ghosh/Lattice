/**
 * useChunkedRecorder
 *
 * Records a MediaStream in 15-second chunks.
 * Each completed chunk is automatically uploaded to POST /upload_video on the
 * vision_server and the returned server-local path is forwarded to the caller
 * via the onChunkReady() callback.
 *
 * Usage:
 *   const { stream, isRecording, chunkCount, requestPermissions, start, stop } =
 *     useChunkedRecorder({ onChunkReady, onError });
 */

'use client';

import { useCallback, useRef, useState } from 'react';

const CHUNK_DURATION_MS = 15_000;          // 15 seconds per chunk
const SERVER_BASE = 'http://localhost:8000';

export interface ChunkedRecorderOptions {
  onChunkReady: (videoPath: string, chunkId: string, chunkIndex: number) => void;
  onError?: (msg: string) => void;
}

export function useChunkedRecorder({ onChunkReady, onError }: ChunkedRecorderOptions) {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [chunkCount, setChunkCount] = useState(0);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunkIndexRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const cycleIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mimeTypeRef = useRef('');

  // ------------------------------------------------------------------
  // Permissions + stream acquisition
  // ------------------------------------------------------------------

  const requestPermissions = useCallback(async (): Promise<MediaStream | null> => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
      streamRef.current = s;
      setStream(s);
      setPermissionGranted(true);
      setPermissionError(null);
      console.log('[ChunkedRecorder] Camera + mic permissions granted');
      return s;
    } catch (err: any) {
      const msg =
        err.name === 'NotAllowedError'
          ? 'Camera/microphone permission denied. Please allow access and refresh.'
          : err.name === 'NotFoundError'
          ? 'No camera or microphone found.'
          : `Permission error: ${err.message}`;
      console.error('[ChunkedRecorder]', msg);
      setPermissionError(msg);
      onError?.(msg);
      return null;
    }
  }, [onError]);

  // ------------------------------------------------------------------
  // Recording lifecycle
  // ------------------------------------------------------------------

  // ------------------------------------------------------------------
  // Create one recorder instance attached to the stream.
  // Each recorder records for CHUNK_DURATION_MS, then stop() is called,
  // which fires ondataavailable with a COMPLETE, self-contained webm blob
  // (no missing EBML header — unlike timeslice continuation blobs).
  // ------------------------------------------------------------------
  const startOneRecorder = useCallback(
    (s: MediaStream, mimeType: string) => {
      const recorder = new MediaRecorder(s, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = async (e: BlobEvent) => {
        if (!e.data || e.data.size < 5000) return; // skip near-empty blobs

        const idx = chunkIndexRef.current++;
        setChunkCount(idx + 1);

        const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
        const filename = `chunk_${String(idx).padStart(4, '0')}.${ext}`;
        const file = new File([e.data], filename, { type: mimeType || 'video/webm' });

        try {
          const form = new FormData();
          form.append('file', file);
          const res = await fetch(`${SERVER_BASE}/upload_video`, { method: 'POST', body: form });
          if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          const json = await res.json();
          console.log(`[ChunkedRecorder] Chunk ${idx} uploaded → ${json.video_path}`);
          onChunkReady(json.video_path, `chunk_${idx}`, idx);
        } catch (err: any) {
          console.error(`[ChunkedRecorder] Chunk ${idx} upload failed:`, err);
          onError?.(`Chunk ${idx} upload failed: ${err.message}`);
        }
      };

      recorder.onerror = (e) => {
        console.error('[ChunkedRecorder] MediaRecorder error:', e);
        onError?.('MediaRecorder error — recording stopped unexpectedly.');
      };

      recorder.start(); // NO timeslice — full blob on stop()
    },
    [onChunkReady, onError]
  );

  const start = useCallback(
    (mediaStream?: MediaStream) => {
      const s = mediaStream ?? streamRef.current;
      if (!s) {
        onError?.('No media stream available. Call requestPermissions() first.');
        return;
      }
      if (cycleIntervalRef.current) return; // already running

      const mimeType =
        ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4'].find(
          (m) => MediaRecorder.isTypeSupported(m)
        ) ?? '';
      mimeTypeRef.current = mimeType;
      chunkIndexRef.current = 0;

      // Start the first recorder immediately
      startOneRecorder(s, mimeType);
      setIsRecording(true);
      console.log(`[ChunkedRecorder] Recording started (${CHUNK_DURATION_MS / 1000}s cycle chunks, mime: ${mimeType || 'default'})`);

      // Every 15 s: stop current recorder (fires ondataavailable with complete blob)
      // then immediately start a fresh one
      cycleIntervalRef.current = setInterval(() => {
        const current = mediaRecorderRef.current;
        if (current && current.state === 'recording') {
          current.stop(); // triggers ondataavailable → upload
        }
        startOneRecorder(s, mimeType); // fresh recorder = fresh EBML header
      }, CHUNK_DURATION_MS);
    },
    [startOneRecorder, onError]
  );

  const stop = useCallback(() => {
    if (cycleIntervalRef.current) {
      clearInterval(cycleIntervalRef.current);
      cycleIntervalRef.current = null;
    }
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === 'recording') {
      recorder.stop(); // final chunk
    }
    setIsRecording(false);
    console.log('[ChunkedRecorder] Recording stopped — final chunk flushing');
  }, []);

  const releaseStream = useCallback(() => {
    stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStream(null);
    setPermissionGranted(false);
    setChunkCount(0);
  }, [stop]);

  return {
    stream,
    isRecording,
    chunkCount,
    permissionGranted,
    permissionError,
    requestPermissions,
    start,
    stop,
    releaseStream,
  };
}
