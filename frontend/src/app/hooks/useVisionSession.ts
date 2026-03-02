/**
 * useVisionSession Hook
 *
 * Manages WebSocket connection to the Python vision server.
 * Supports both:
 *  - Full-session mode: startSession(videoPath) / stopSession()
 *  - Per-chunk mode:    processChunk(videoPath, chunkId)  ← used for live 15-s recording
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface GazeLogEntry {
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

export interface PredictionSummary {
  count: number;
  mean_confidence: number;
  min_confidence: number;
  max_confidence: number;
}

export interface VoiceAnalysis {
  score: number | null;
  window_times: number[];
  window_scores: number[];
  error: string | null;
}

export interface ChunkResult {
  chunkId: string;
  chunkIndex: number;
  gaze_data: GazeLogEntry[];
  predictions: PredictionEntry[];
  inference_summary: PredictionSummary | null;
  voice_analysis: VoiceAnalysis | null;
  facial_analysis: { score: number | null; error: string | null } | null;
  receivedAt: string;
}

export interface SessionData {
  session_id: string;
  log_data: GazeLogEntry[];
  predictions: PredictionEntry[];
  summary: PredictionSummary;
  voice_analysis: VoiceAnalysis | null;
  start_time: string;
  end_time: string;
}

interface SessionStatus {
  is_running: boolean;
  session_id?: string;
}

export function useVisionSession() {
  const [isConnected, setIsConnected] = useState(false);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Real-time prediction state (full-session mode)
  const [predictions, setPredictions] = useState<PredictionEntry[]>([]);
  const [predictionSummary, setPredictionSummary] = useState<PredictionSummary | null>(null);
  const [latestConfidence, setLatestConfidence] = useState<number | null>(null);
  const [latestFacialScore, setLatestFacialScore] = useState<number | null>(null);

  // Per-chunk results (chunked recording mode)
  const [chunkResults, setChunkResults] = useState<ChunkResult[]>([]);
  const [latestVoiceScore, setLatestVoiceScore] = useState<number | null>(null);
  const [processingChunks, setProcessingChunks] = useState<Set<string>>(new Set());
  const [chunkErrors, setChunkErrors] = useState<string[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  // Outbound message queue — holds messages we tried to send while WS was not open
  const outboundQueueRef = useRef<string[]>([]);
  // Per-chunk timeout handles — cleared when chunk_processed/chunk_error arrives
  const chunkTimeoutsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());

  const _sendOrQueue = useCallback((payload: object) => {
    const json = JSON.stringify(payload);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(json);
    } else {
      console.warn('[useVisionSession] WS not open — queuing message');
      outboundQueueRef.current.push(json);
    }
  }, []);

  const _clearChunkTimeout = useCallback((chunkId: string) => {
    const handle = chunkTimeoutsRef.current.get(chunkId);
    if (handle) {
      clearTimeout(handle);
      chunkTimeoutsRef.current.delete(chunkId);
    }
  }, []);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('✓ Connected to vision server');
        setIsConnected(true);
        setError(null);
        // Flush any queued messages
        const queue = outboundQueueRef.current.splice(0);
        queue.forEach((msg) => ws.send(msg));
        if (queue.length) console.log(`[useVisionSession] Flushed ${queue.length} queued message(s)`);
      };
      
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
          case 'session_started':
            console.log('✓ Session started:', message.session_id);
            setCurrentSessionId(message.session_id);
            setIsSessionActive(true);
            setSessionData(null);
            setPredictions([]);
            setPredictionSummary(null);
            setLatestConfidence(null);
            break;
            
          case 'prediction_update':
            console.log('📊 New prediction:', message.prediction);
            setPredictions(prev => [...prev, message.prediction]);
            setLatestConfidence(message.prediction.confidence);
            break;
            
          case 'prediction_summary':
            console.log('📈 Prediction summary:', message.summary);
            setPredictionSummary(message.summary);
            break;
            
          case 'session_ended':
            console.log('✓ Session ended:', message.session_id);
            console.log('  Log entries:', message.log_data.length);
            console.log('  Predictions:', message.predictions?.length || 0);
            if (message.voice_analysis?.score != null) {
              console.log('  Voice score:', message.voice_analysis.score);
              setLatestVoiceScore(message.voice_analysis.score);
            }
            setSessionData(message);
            setPredictions(message.predictions || []);
            setPredictionSummary(message.summary || null);
            setIsSessionActive(false);
            setCurrentSessionId(null);
            break;

          case 'chunk_processed': {
            _clearChunkTimeout(message.chunk_id);
            // Strip bulky arrays (energy, pitch_proxy) from voice_analysis before storing
            const rawVoice = message.voice_analysis;
            const voice_analysis: VoiceAnalysis | null = rawVoice
              ? {
                  score: rawVoice.score ?? null,
                  window_times: rawVoice.window_times ?? [],
                  window_scores: rawVoice.window_scores ?? [],
                  error: rawVoice.error ?? null,
                }
              : null;
            const result: ChunkResult = {
              chunkId: message.chunk_id,
              chunkIndex: message.chunk_index ?? 0,
              gaze_data: message.gaze_data || [],
              predictions: message.predictions || [],
              inference_summary: message.inference_summary || null,
              voice_analysis,
              facial_analysis: message.facial_analysis || null,
              receivedAt: new Date().toISOString(),
            };
            console.log(
              `📦 Chunk processed: ${result.chunkId} | voice=${result.voice_analysis?.score?.toFixed(3)} | confidence=${result.predictions[0]?.confidence?.toFixed(3)} | facial=${result.facial_analysis?.score?.toFixed(3)} | gaze=${result.gaze_data.length} events`
            );
            setChunkResults((prev) => [...prev, result]);
            if (result.voice_analysis?.score != null) {
              setLatestVoiceScore(result.voice_analysis.score);
            }
            if (result.predictions.length > 0) {
              const lastConf = result.predictions[result.predictions.length - 1].confidence;
              setLatestConfidence(lastConf);
            }
            if (result.facial_analysis?.score != null) {
              setLatestFacialScore(result.facial_analysis.score);
            }
            setProcessingChunks((prev) => {
              const next = new Set(prev);
              next.delete(message.chunk_id);
              return next;
            });
            break;
          }

          case 'chunk_error':
            console.error('Chunk processing error:', message.chunk_id, message.message);
            _clearChunkTimeout(message.chunk_id);
            setChunkErrors((prev) => [...prev, `Chunk ${message.chunk_id?.slice(0, 8)}: ${message.message ?? 'unknown error'}`]);
            setProcessingChunks((prev) => {
              const next = new Set(prev);
              next.delete(message.chunk_id);
              return next;
            });
            break;

          case 'error':
            console.error('Server error:', message.message);
            setError(message.message);
            break;

          case 'status':
            setIsSessionActive(message.is_running);
            if (message.session_id) {
              setCurrentSessionId(message.session_id);
            }
            break;

          default:
            console.log('Unknown message type:', message.type);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Failed to connect to vision server. Is vision_server.py running?');
      };
      
      ws.onclose = () => {
        console.log('Disconnected from vision server');
        setIsConnected(false);
        setIsSessionActive(false);
        
        // Attempt reconnection after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      };
      
      wsRef.current = ws;
      
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to connect to vision server');
    }
  }, [_clearChunkTimeout]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  /**
   * Start a full-video session. videoPath must be the server-local path returned
   * by POST /upload_video. Used for post-interview analysis of the full recording.
   */
  const startSession = useCallback((videoPath: string) => {
    _sendOrQueue({ action: 'start_session', video_path: videoPath });
  }, [_sendOrQueue]);

  /**
   * Send a 15-s video chunk to the server for parallel processing:
   * vision.py (gaze) + VideoMAE (confidence) + FacialExpression + voice_analyzer.
   */
  const processChunk = useCallback(
    (videoPath: string, chunkId: string, chunkIndex: number = 0) => {
      setProcessingChunks((prev) => new Set(prev).add(chunkId));

      // 2-minute timeout — if server never responds, clear the spinner
      const timeoutHandle = setTimeout(() => {
        console.warn(`[useVisionSession] Chunk ${chunkId} timed out after 120s`);
        setProcessingChunks((prev) => {
          const next = new Set(prev);
          next.delete(chunkId);
          return next;
        });
        chunkTimeoutsRef.current.delete(chunkId);
      }, 120_000);
      chunkTimeoutsRef.current.set(chunkId, timeoutHandle);

      _sendOrQueue({
        action: 'process_chunk',
        video_path: videoPath,
        chunk_id: chunkId,
        chunk_index: chunkIndex,
      });
      console.log(`[useVisionSession] process_chunk sent/queued: ${chunkId} → ${videoPath}`);
    },
    [_sendOrQueue]
  );

  const stopSession = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return;
    }
    
    wsRef.current.send(JSON.stringify({ action: 'stop_session' }));
  }, []);

  const getStatus = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    
    wsRef.current.send(JSON.stringify({ action: 'get_status' }));
  }, []);

  const clearChunkErrors = useCallback(() => setChunkErrors([]), []);
  const dismissChunkError = useCallback(
    (index: number) => setChunkErrors((prev) => prev.filter((_, i) => i !== index)),
    []
  );

  // Auto-connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    // Connection state
    isConnected,
    error,

    // Full-session state
    isSessionActive,
    currentSessionId,
    sessionData,

    // Per-chunk state
    chunkResults,
    latestVoiceScore,
    latestFacialScore,
    processingChunks,
    pendingChunks: processingChunks.size,
    chunkErrors,
    clearChunkErrors,
    dismissChunkError,
    
    // Real-time prediction state
    predictions,
    predictionSummary,
    latestConfidence,
    
    // Actions
    startSession,
    stopSession,
    processChunk,
    getStatus,
    reconnect: connect,
  };
}
