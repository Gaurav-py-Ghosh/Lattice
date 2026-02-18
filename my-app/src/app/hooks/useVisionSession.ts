/**
 * useVisionSession Hook
 * 
 * Manages WebSocket connection to the Python vision server
 * Handles session lifecycle (start/stop) and gaze log data retrieval
 */

import { useCallback, useEffect, useRef, useState } from 'react';

interface GazeLogEntry {
  timestamp: string;
  status: string;
}

interface PredictionEntry {
  chunk: number;
  video_file: string;
  confidence: number;
  timestamp: number;
  processing_time: number;
}

interface PredictionSummary {
  count: number;
  mean_confidence: number;
  min_confidence: number;
  max_confidence: number;
}

interface SessionData {
  session_id: string;
  log_data: GazeLogEntry[];
  predictions: PredictionEntry[];
  summary: PredictionSummary;
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
  
  // Real-time prediction state
  const [predictions, setPredictions] = useState<PredictionEntry[]>([]);
  const [predictionSummary, setPredictionSummary] = useState<PredictionSummary | null>(null);
  const [latestConfidence, setLatestConfidence] = useState<number | null>(null);
  
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
            setSessionData(message);
            setPredictions(message.predictions || []);
            setPredictionSummary(message.summary || null);
            setIsSessionActive(false);
            setCurrentSessionId(null);
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

  const startSession = useCallback((headless: boolean = true) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return;
    }
    
    wsRef.current.send(JSON.stringify({ 
      action: 'start_session',
      headless: headless // true = no windows, false = minimized windows
    }));
  }, []);

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
    
    // Session state
    isSessionActive,
    currentSessionId,
    sessionData,
    
    // Real-time prediction state
    predictions,
    predictionSummary,
    latestConfidence,
    
    // Actions
    startSession,
    stopSession,
    getStatus,
    reconnect: connect,
  };
}
