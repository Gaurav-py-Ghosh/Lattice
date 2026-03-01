'use client';

import { GazeData } from '../hooks/useGazeTracking';

interface GazeDebugPanelProps {
  gazeData: GazeData | null;
  fps: number;
  connected: boolean;
  calibrated: boolean;
  screenCalibrated: boolean;
  isOpen: boolean;
  onClose: () => void;
  onCalibrate: () => void;
  onAddMarker: () => void;
  onReset: () => void;
}

export default function GazeDebugPanel({
  gazeData,
  fps,
  connected,
  calibrated,
  screenCalibrated,
  isOpen,
  onClose,
  onCalibrate,
  onAddMarker,
  onReset,
}: GazeDebugPanelProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-6">
      <div className="bg-[#0a0a0f] border border-white/20 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <svg className="w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Gaze Tracking Debug Panel
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
          
          {/* Connection Status */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white/5 border border-white/10 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Connection</span>
                <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              </div>
              <p className={`text-lg font-bold mt-1 ${connected ? 'text-green-400' : 'text-red-400'}`}>
                {connected ? 'Connected' : 'Disconnected'}
              </p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">FPS</span>
                <div className={`w-2 h-2 rounded-full ${fps > 20 ? 'bg-green-400' : fps > 10 ? 'bg-yellow-400' : 'bg-red-400'}`} />
              </div>
              <p className={`text-lg font-bold mt-1 ${fps > 20 ? 'text-green-400' : fps > 10 ? 'text-yellow-400' : 'text-red-400'}`}>
                {fps} FPS
              </p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-xl p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Calibrated</span>
                <div className={`w-2 h-2 rounded-full ${calibrated ? 'bg-green-400' : 'bg-gray-400'}`} />
              </div>
              <p className={`text-lg font-bold mt-1 ${calibrated ? 'text-green-400' : 'text-gray-400'}`}>
                {calibrated ? 'Yes' : 'No'}
              </p>
            </div>
          </div>

          {/* Gaze Data */}
          {gazeData && (
            <>
              {/* Real-time Values */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Real-time Data</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-white/5 border border-white/10 rounded-lg p-3">
                    <span className="text-xs text-gray-400 block mb-1">Screen X</span>
                    <span className="text-lg font-mono font-bold text-purple-400">{gazeData.screen_x}px</span>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-lg p-3">
                    <span className="text-xs text-gray-400 block mb-1">Screen Y</span>
                    <span className="text-lg font-mono font-bold text-purple-400">{gazeData.screen_y}px</span>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-lg p-3">
                    <span className="text-xs text-gray-400 block mb-1">Markers</span>
                    <span className="text-lg font-mono font-bold text-blue-400">{gazeData.marker_count}</span>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-lg p-3">
                    <span className="text-xs text-gray-400 block mb-1">Timestamp</span>
                    <span className="text-xs font-mono text-gray-300">{new Date(gazeData.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>

              {/* Status Indicators */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Status</h3>
                <div className="space-y-2">
                  <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Gaze Status</span>
                      <span className={`text-sm font-semibold ${
                        gazeData.gaze_status === 'Looking Forward' ? 'text-green-400' :
                        gazeData.gaze_status === 'No Face Detected' ? 'text-gray-400' :
                        'text-yellow-400'
                      }`}>
                        {gazeData.gaze_status}
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Eye Status</span>
                      <span className={`text-sm font-semibold ${
                        gazeData.eye_status === 'Eyes Centered (Head Ref)' ? 'text-green-400' :
                        'text-yellow-400'
                      }`}>
                        {gazeData.eye_status}
                      </span>
                    </div>
                  </div>

                  <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Screen Calibration</span>
                      <span className={`text-sm font-semibold ${screenCalibrated ? 'text-green-400' : 'text-gray-400'}`}>
                        {screenCalibrated ? 'Applied' : 'Not Applied'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Visualization */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Gaze Visualization</h3>
                <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-white/10 rounded-lg overflow-hidden relative h-64">
                  <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:32px_32px]" />
                  
                  {/* Crosshair at center */}
                  <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2">
                    <div className="w-8 h-0.5 bg-white/30"></div>
                  </div>
                  <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2">
                    <div className="w-0.5 h-8 bg-white/30"></div>
                  </div>

                  {/* Gaze point */}
                  {gazeData.screen_x >= 0 && gazeData.screen_y >= 0 && (
                    <div
                      className="absolute w-4 h-4 transform -translate-x-1/2 -translate-y-1/2"
                      style={{
                        left: `${(gazeData.screen_x / window.screen.width) * 100}%`,
                        top: `${(gazeData.screen_y / window.screen.height) * 100}%`,
                      }}
                    >
                      <div className="w-full h-full bg-purple-500 rounded-full shadow-lg shadow-purple-500/50 animate-pulse" />
                      <div className="absolute inset-0 w-full h-full bg-purple-500/30 rounded-full animate-ping" />
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* Controls */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white mb-3">Controls</h3>
            
            <button
              onClick={onCalibrate}
              disabled={!calibrated}
              className="w-full py-3 bg-purple-500/20 hover:bg-purple-500/30 disabled:bg-white/5 disabled:text-gray-500 text-purple-400 font-medium rounded-lg transition-all border border-purple-500/30 disabled:border-white/5"
            >
              Calibrate Screen
            </button>

            <button
              onClick={onAddMarker}
              disabled={!calibrated}
              className="w-full py-3 bg-blue-500/20 hover:bg-blue-500/30 disabled:bg-white/5 disabled:text-gray-500 text-blue-400 font-medium rounded-lg transition-all border border-blue-500/30 disabled:border-white/5"
            >
              Add Marker at Gaze Point
            </button>

            <button
              onClick={onReset}
              className="w-full py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 font-medium rounded-lg transition-all border border-red-500/30"
            >
              Reset Calibration
            </button>
          </div>

          {/* Info */}
          <div className="mt-6 bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <svg className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="text-sm font-medium text-blue-400 mb-1">Debug Mode</p>
                <p className="text-sm text-gray-400">
                  This panel shows real-time gaze tracking data from the Vision Server. 
                  Use calibration controls to improve accuracy. Close this panel when ready to interview.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
