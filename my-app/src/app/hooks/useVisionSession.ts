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

interface SessionData {
  session_id: string;
  log_data: GazeLogEntry[];
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
            break;
            
          case 'session_ended':
            console.log('✓ Session ended:', message.session_id);
            console.log('  Log entries:', message.log_data.length);
            setSessionData(message);
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

  const startSession = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return;
    }
    
    wsRef.current.send(JSON.stringify({ action: 'start_session' }));
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
    
    // Actions
    startSession,
    stopSession,
    getStatus,
    reconnect: connect,
  };
}
