/**
 * VisionSessionControl Component
 * 
 * Provides UI controls to start/stop vision.py sessions
 * and displays gaze tracking results when session ends
 */

'use client';

import React, { useEffect, useState } from 'react';
import { useVisionSession } from '../hooks/useVisionSession';

interface VisionSessionControlProps {
  onSessionEnd?: (logData: any) => void;
  autoStart?: boolean;
  headless?: boolean;  // true = no windows (default), false = show minimized windows
}

export default function VisionSessionControl({ 
  onSessionEnd, 
  autoStart = false,
  headless = true  // Default to headless mode (no OpenCV windows)
}: VisionSessionControlProps) {
  const {
    isConnected,
    error,
    isSessionActive,
    currentSessionId,
    sessionData,
    startSession,
    stopSession,
    reconnect,
  } = useVisionSession();

  const [showResults, setShowResults] = useState(false);

  // Callback when session ends
  useEffect(() => {
    if (sessionData && !isSessionActive) {
      setShowResults(true);
      if (onSessionEnd) {
        onSessionEnd(sessionData);
      }
    }
  }, [sessionData, isSessionActive, onSessionEnd]);

  // Auto-start if enabled
  useEffect(() => {
    if (autoStart && isConnected && !isSessionActive) {
      startSession(headless);
    }
  }, [autoStart, isConnected, isSessionActive, startSession, headless]);

  const handleStartSession = () => {
    setShowResults(false);
    startSession(headless);
  };

  const handleStopSession = () => {
    stopSession();
  };

  const calculateGazeStats = () => {
    if (!sessionData?.log_data) return null;

    const total = sessionData.log_data.length;
    const lookingAwayCount = sessionData.log_data.filter(
      entry => entry.status.includes('Looking Away') || entry.status.includes('Eyes Away')
    ).length;
    const focusedCount = total - lookingAwayCount;

    return {
      total,
      focusedCount,
      lookingAwayCount,
      focusPercentage: total > 0 ? ((focusedCount / total) * 100).toFixed(1) : '0',
    };
  };

  const stats = sessionData ? calculateGazeStats() : null;

  return (
    <div className="vision-session-control">
      {/* Connection Status */}
      <div className="mb-4 flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
        <span className="text-sm text-gray-600">
          {isConnected ? 'Connected to Vision Server' : 'Disconnected'}
        </span>
        {!isConnected && (
          <button
            onClick={reconnect}
            className="ml-2 text-xs px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Reconnect
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">{error}</p>
          <p className="text-xs text-red-500 mt-1">
            Make sure you run: <code className="bg-red-100 px-1">conda activate pupil310 && cd Vision && python vision_server.py</code>
          </p>
        </div>
      )}

      {/* Session Controls */}
      <div className="mb-6">
        {!isSessionActive ? (
          <button
            onClick={handleStartSession}
            disabled={!isConnected}
            className={`px-6 py-3 rounded-lg font-semibold text-white transition-all
              ${isConnected 
                ? 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-lg hover:shadow-xl' 
                : 'bg-gray-400 cursor-not-allowed'
              }`}
          >
            🎥 Start Gaze Tracking Session
          </button>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm font-medium text-gray-700">
                Recording Session: {currentSessionId?.slice(0, 8)}...
              </span>
            </div>
            <button
              onClick={handleStopSession}
              className="px-6 py-3 rounded-lg font-semibold text-white bg-red-500 hover:bg-red-600 transition-all shadow-lg hover:shadow-xl"
            >
              ⏹️ Stop Recording
            </button>
          </div>
        )}
      </div>

      {/* Session Results */}
      {showResults && sessionData && stats && (
        <div className="mt-6 p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl shadow-lg">
          <h3 className="text-xl font-bold text-gray-800 mb-4">📊 Session Results</h3>
          
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <div className="text-2xl font-bold text-blue-600">{stats.focusPercentage}%</div>
              <div className="text-xs text-gray-600 mt-1">Focus Score</div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <div className="text-2xl font-bold text-green-600">{stats.focusedCount}</div>
              <div className="text-xs text-gray-600 mt-1">Focused</div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <div className="text-2xl font-bold text-red-600">{stats.lookingAwayCount}</div>
              <div className="text-xs text-gray-600 mt-1">Looking Away</div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <div className="text-2xl font-bold text-purple-600">{stats.total}</div>
              <div className="text-xs text-gray-600 mt-1">Total Events</div>
            </div>
          </div>

          {/* Session Info */}
          <div className="bg-white p-4 rounded-lg shadow mb-4">
            <div className="text-xs text-gray-500 space-y-1">
              <div>Session ID: <span className="font-mono text-gray-700">{sessionData.session_id}</span></div>
              <div>Started: <span className="font-mono text-gray-700">{new Date(sessionData.start_time).toLocaleString()}</span></div>
              <div>Ended: <span className="font-mono text-gray-700">{new Date(sessionData.end_time).toLocaleString()}</span></div>
            </div>
          </div>

          {/* Gaze Timeline (scrollable) */}
          <div className="bg-white p-4 rounded-lg shadow max-h-64 overflow-y-auto">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Gaze Timeline</h4>
            <div className="space-y-2">
              {sessionData.log_data.slice(-20).reverse().map((entry, idx) => (
                <div 
                  key={idx} 
                  className={`text-xs p-2 rounded ${
                    entry.status.includes('Looking Away') || entry.status.includes('Eyes Away')
                      ? 'bg-red-50 border-l-2 border-red-400'
                      : 'bg-green-50 border-l-2 border-green-400'
                  }`}
                >
                  <span className="font-mono text-gray-500">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="ml-3 font-medium text-gray-700">
                    {entry.status}
                  </span>
                </div>
              ))}
            </div>
            {sessionData.log_data.length > 20 && (
              <p className="text-xs text-gray-500 mt-2 text-center">
                Showing last 20 of {sessionData.log_data.length} entries
              </p>
            )}
          </div>
        </div>
      )}

      <style jsx>{`
        .vision-session-control {
          max-width: 100%;
        }

        code {
          font-family: 'Courier New', monospace;
          font-size: 0.85em;
          padding: 2px 4px;
          border-radius: 3px;
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
}
