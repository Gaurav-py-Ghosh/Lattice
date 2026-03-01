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
  energy: number[];
  pitch_proxy: number[];
  error: string | null;
}

export interface ChunkResult {
  chunkId: string;
  chunkIndex: number;
  gaze_data: GazeLogEntry[];
  predictions: PredictionEntry[];
  inference_summary: PredictionSummary | null;
  voice_analysis: VoiceAnalysis | null;
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

  // Per-chunk results (chunked recording mode)
  const [chunkResults, setChunkResults] = useState<ChunkResult[]>([]);
  const [latestVoiceScore, setLatestVoiceScore] = useState<number | null>(null);
  const [processingChunks, setProcessingChunks] = useState<Set<string>>(new Set());

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('✓ Connected to vision server');
        setIsConnected(true);
        setError(null);
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
            const result: ChunkResult = {
              chunkId: message.chunk_id,
              chunkIndex: message.chunk_index ?? 0,
              gaze_data: message.gaze_data || [],
              predictions: message.predictions || [],
              inference_summary: message.inference_summary || null,
              voice_analysis: message.voice_analysis || null,
              receivedAt: new Date().toISOString(),
            };
            console.log(
              `📦 Chunk processed: ${result.chunkId} | voice=${result.voice_analysis?.score?.toFixed(3)} | gaze=${result.gaze_data.length} events`
            );
            setChunkResults((prev) => [...prev, result]);
            if (result.voice_analysis?.score != null) {
              setLatestVoiceScore(result.voice_analysis.score);
            }
            if (result.predictions.length > 0) {
              const lastConf = result.predictions[result.predictions.length - 1].confidence;
              setLatestConfidence(lastConf);
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
  }, []);

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
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return;
    }
    wsRef.current.send(JSON.stringify({
      action: 'start_session',
      video_path: videoPath,
    }));
  }, []);

  /**
   * Send a 15-s video chunk to the server for parallel processing:
   * vision.py (gaze) + realtime_inference (confidence) + voice_analyzer.
   */
  const processChunk = useCallback(
    (videoPath: string, chunkId: string, chunkIndex: number = 0) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setError('Not connected to server');
        return;
      }
      setProcessingChunks((prev) => new Set(prev).add(chunkId));
      wsRef.current.send(
        JSON.stringify({
          action: 'process_chunk',
          video_path: videoPath,
          chunk_id: chunkId,
          chunk_index: chunkIndex,
        })
      );
      console.log(`[useVisionSession] process_chunk sent: ${chunkId} → ${videoPath}`);
    },
    []
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
    processingChunks,
    pendingChunks: processingChunks.size,
    
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
