import { useEffect, useRef, useState, useCallback } from 'react';

export interface GazeData {
  screen_x: number;
  screen_y: number;
  gaze_status: string;
  eye_status: string;
  calibrated: boolean;
  screen_calibrated: boolean;
  marker_count: number;
  timestamp: string;
  command_result?: {
    success: boolean;
    message: string;
  };
}

export interface UseGazeTrackingReturn {
  gazeData: GazeData | null;
  connected: boolean;
  calibrated: boolean;
  screenCalibrated: boolean;
  fps: number;
  connect: () => void;
  disconnect: () => void;
  startTracking: (videoElement: HTMLVideoElement) => void;
  stopTracking: () => void;
  calibrateScreen: () => void;
  addMarker: () => void;
  resetCalibration: () => void;
  isTracking: boolean;
}

export const useGazeTracking = (
  serverUrl: string = 'ws://localhost:8000'
): UseGazeTrackingReturn => {
  const [gazeData, setGazeData] = useState<GazeData | null>(null);
  const [connected, setConnected] = useState(false);
  const [calibrated, setCalibrated] = useState(false);
  const [screenCalibrated, setScreenCalibrated] = useState(false);
  const [fps, setFps] = useState(0);
  const [isTracking, setIsTracking] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const sendIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Connect to WebSocket server
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[Gaze] Already connected');
      return;
    }

    console.log('[Gaze] Connecting to', serverUrl);
    const ws = new WebSocket(serverUrl);

    ws.onopen = () => {
      console.log('✅ [Gaze] Connected to Vision Server');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: GazeData = JSON.parse(event.data);
        setGazeData(data);
        setCalibrated(data.calibrated);
        setScreenCalibrated(data.screen_calibrated);

        // Update FPS
        frameCountRef.current++;
        const now = Date.now();
        if (now - lastFpsUpdateRef.current >= 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }

        // Log command results
        if (data.command_result) {
          console.log(
            data.command_result.success ? '✅' : '❌',
            '[Gaze]',
            data.command_result.message
          );
        }
      } catch (error) {
        console.error('[Gaze] Error parsing data:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('❌ [Gaze] WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('🔌 [Gaze] Disconnected from Vision Server');
      setConnected(false);
      setCalibrated(false);
      setScreenCalibrated(false);
      setIsTracking(false);
    };

    wsRef.current = ws;
  }, [serverUrl]);

  // Disconnect from server
  const disconnect = useCallback(() => {
    stopTracking();
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  // Start sending frames from video element
  const startTracking = useCallback((videoElement: HTMLVideoElement) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('[Gaze] Not connected to server');
      return;
    }

    if (isTracking) {
      console.log('[Gaze] Already tracking');
      return;
    }

    videoRef.current = videoElement;
    setIsTracking(true);

    // Send frames at 30 FPS
    sendIntervalRef.current = setInterval(() => {
      if (
        !videoRef.current ||
        !wsRef.current ||
        wsRef.current.readyState !== WebSocket.OPEN
      ) {
        return;
      }

      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        canvas.toBlob(
          (blob) => {
            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(blob);
            }
          },
          'image/jpeg',
          0.8
        );
      }
    }, 33); // ~30 FPS

    console.log('📹 [Gaze] Started tracking');
  }, [isTracking]);

  // Stop sending frames
  const stopTracking = useCallback(() => {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    videoRef.current = null;
    setIsTracking(false);
    console.log('📹 [Gaze] Stopped tracking');
  }, []);

  // Send command to server
  const sendCommand = useCallback((action: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('[Gaze] Not connected to server');
      return;
    }

    wsRef.current.send(JSON.stringify({ action }));
    console.log('[Gaze] Sent command:', action);
  }, []);

  // Calibrate screen
  const calibrateScreen = useCallback(() => {
    sendCommand('calibrate_screen');
  }, [sendCommand]);

  // Add marker
  const addMarker = useCallback(() => {
    sendCommand('add_marker');
  }, [sendCommand]);

  // Reset calibration
  const resetCalibration = useCallback(() => {
    sendCommand('reset_calibration');
    setScreenCalibrated(false);
  }, [sendCommand]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    gazeData,
    connected,
    calibrated,
    screenCalibrated,
    fps,
    connect,
    disconnect,
    startTracking,
    stopTracking,
    calibrateScreen,
    addMarker,
    resetCalibration,
    isTracking,
  };
};
