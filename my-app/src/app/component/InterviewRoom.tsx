'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useVisionSession } from '../hooks/useVisionSession';
import CalibrationFlow from './CalibrationFlow';

export default function InterviewRoom() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [showCalibration, setShowCalibration] = useState(true);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [visionSessionData, setVisionSessionData] = useState<any>(null);
  const [showResults, setShowResults] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Vision.py session hook
  const {
    isConnected: visionConnected,
    isSessionActive: visionActive,
    sessionData,
    predictions,
    predictionSummary,
    latestConfidence,
    startSession: startVisionSession,
    stopSession: stopVisionSession,
    error: visionError,
  } = useVisionSession();

  // CAMERA IS HANDLED BY VISION.PY - Don't initialize here to avoid conflicts
  // Vision.py will have exclusive camera access with its debug window
  useEffect(() => {
    // Just set a dummy stream to satisfy video element requirements
    console.log('📷 Camera will be handled by vision.py window');

    return () => {
      // No cleanup needed - vision.py handles camera
    };
  }, []);

  // Vision.py handles gaze tracking - we don't use the old gaze tracking hook
  // useEffect(() => {
  //   if (stream && !connected) {
  //     connect();
  //   }
  //   return () => {
  //     disconnect();
  //   };
  // }, [stream, connected, connect, disconnect]);

  // Vision.py handles tracking - no need to start/stop from website
  // useEffect(() => {
  //   if (interviewStarted && videoRef.current && connected) {
  //     startTracking(videoRef.current);
  //   } else if (!interviewStarted && isTracking) {
  //     stopTracking();
  //   }
  //   return () => {
  //     stopTracking();
  //   };
  // }, [interviewStarted, connected, startTracking, stopTracking, isTracking]);

  // Recording timer
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isRecording) {
      timer = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      setRecordingTime(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isRecording]);

  // Handle vision session end
  useEffect(() => {
    if (sessionData && !visionActive) {
      console.log('📊 Vision session ended, got data:', sessionData);
      setVisionSessionData(sessionData);
      setShowResults(true);
    }
  }, [sessionData, visionActive]);

  // Handle fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  const enterFullscreen = async () => {
    if (containerRef.current && !document.fullscreenElement) {
      try {
        await containerRef.current.requestFullscreen();
        console.log('✅ Entered fullscreen mode');
      } catch (err) {
        console.error('❌ Error entering fullscreen:', err);
      }
    }
  };

  const exitFullscreen = async () => {
    if (document.fullscreenElement) {
      try {
        await document.exitFullscreen();
        console.log('✅ Exited fullscreen mode');
      } catch (err) {
        console.error('❌ Error exiting fullscreen:', err);
      }
    }
  };

  const handleCalibrationComplete = () => {
    setShowCalibration(false);
    setInterviewStarted(true);
    
    // Enter fullscreen mode
    enterFullscreen();
    
    // Start vision.py session when interview starts
    if (visionConnected && !visionActive) {
      console.log('🎥 Starting vision.py session in headless mode (no windows)...');
      startVisionSession(true); // headless=true: no OpenCV windows
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleLeave = async () => {
    if (confirm('Leave the interview? Your progress will be saved.')) {
      // Exit fullscreen first
      await exitFullscreen();
      
      // Stop vision session and wait for results
      if (visionActive) {
        console.log('⏹️ Stopping vision.py session...');
        stopVisionSession();
      }
      
      // Wait a moment for session data to arrive
      setTimeout(() => {
        router.push('/');
      }, 1000);
    }
  };

  return (
    <div ref={containerRef} className="min-h-screen bg-[#0a0a0f] text-white flex flex-col">
      {/* Calibration Flow */}
      {showCalibration && (
        <CalibrationFlow
          onComplete={handleCalibrationComplete}
          onCalibrate={() => console.log('Calibration handled by vision.py')}
          calibrated={true}
          screenCalibrated={true}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center font-bold text-sm">
            AI
          </div>
          <span className="text-xl font-bold tracking-tight">InterviewAR</span>
          {isRecording && (
            <div className="flex items-center gap-2 px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-full">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm text-red-400 font-mono">{formatTime(recordingTime)}</span>
            </div>
          )}
          {/* Vision.py Session Indicator */}
          {visionActive && (
            <div className="flex items-center gap-2 px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded-full">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
              <span className="text-sm text-purple-400 font-mono">👁️ Gaze Tracking</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-4">
          {/* Fullscreen Toggle */}
          {isFullscreen ? (
            <button
              onClick={exitFullscreen}
              className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-sm font-medium rounded-lg transition-all border border-blue-500/30 flex items-center gap-2"
              title="Exit Fullscreen"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Exit Fullscreen
            </button>
          ) : interviewStarted && (
            <button
              onClick={enterFullscreen}
              className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-sm font-medium rounded-lg transition-all border border-blue-500/30 flex items-center gap-2"
              title="Enter Fullscreen"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
              Fullscreen
            </button>
          )}
          {/* Vision Server Status */}
          {visionError && (
            <div className="px-3 py-1.5 bg-red-500/20 border border-red-500/30 rounded-lg text-xs text-red-400">
              Vision Server Offline
            </div>
          )}
          {visionConnected && !visionError && (
            <div className="flex items-center gap-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full" />
              <span className="text-green-400">Vision Server</span>
            </div>
          )}
          {/* Manual Vision Control */}
          {!visionActive && visionConnected && interviewStarted && (
            <button
              onClick={() => startVisionSession(true)} // headless=true
              className="px-3 py-1.5 bg-green-500/20 hover:bg-green-500/30 text-green-400 text-sm font-medium rounded-lg transition-all border border-green-500/30"
            >
              Start Gaze Track
            </button>
          )}
          {visionActive && (
            <button
              onClick={stopVisionSession}
              className="px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-sm font-medium rounded-lg transition-all border border-red-500/30"
            >
              Stop Gaze Track
            </button>
          )}


          
          <button 
            onClick={() => {
              if (confirm('Are you sure you want to end this interview?')) {
                handleLeave();
              }
            }}
            className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
          >
            End Interview
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center p-6 relative">
        {/* Background Effects */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] opacity-50" />
        </div>

        {/* Video Container */}
        <div className="relative w-full max-w-5xl aspect-video">
          {/* Main Video Feed - Managed by Vision.py */}
          <div className="relative w-full h-full bg-gradient-to-br from-gray-900 via-purple-900/20 to-gray-900 rounded-2xl overflow-hidden border-2 border-white/10 shadow-2xl">
            
            {/* Vision.py Window Indicator */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center max-w-md p-8">
                {visionActive ? (
                  <>
                    <div className="w-32 h-32 mx-auto mb-6 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center border-4 border-purple-500/30 animate-pulse">
                      <svg className="w-16 h-16 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-3">
                      👁️ Gaze Tracking Active
                    </h3>
                    <p className="text-gray-400 mb-4">
                      Check the <span className="text-purple-400 font-semibold">Vision.py window</span> for camera feed and calibration
                    </p>
                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 text-sm text-gray-300">
                      <p className="mb-2">
                        <span className="text-purple-400 font-mono">Press 'C'</span> in vision window to calibrate
                      </p>
                      <p>
                        <span className="text-purple-400 font-mono">Press 'X'</span> to add markers
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-32 h-32 mx-auto mb-6 bg-white/5 rounded-full flex items-center justify-center border-4 border-white/10">
                      <svg className="w-16 h-16 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        <line x1="3" y1="3" x2="21" y2="21" stroke="currentColor" strokeWidth="2"/>
                      </svg>
                    </div>
                    <h3 className="text-xl font-bold text-white mb-3">
                      Camera Not Active
                    </h3>
                    <p className="text-gray-400">
                      Complete calibration to start vision tracking
                    </p>
                  </>
                )}
              </div>
            </div>

            {/* Tracking Status Badge */}
            <div className="absolute top-4 right-4 px-3 py-2 bg-gradient-to-br from-purple-500/80 to-pink-500/80 backdrop-blur-sm rounded-lg border border-white/30">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                  <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"/>
                </svg>
                <span className="text-sm font-medium">
                  {interviewStarted ? 'AI Monitoring' : 'Calibrating'}
                </span>
              </div>
            </div>

            {/* Interview Question Area (bottom overlay) */}
            {interviewStarted && (
              <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
                <div className="max-w-3xl mx-auto">
                  <p className="text-sm text-purple-400 mb-2">Current Question</p>
                  <p className="text-lg text-white">
                    Tell me about a time when you had to work under pressure. How did you handle it?
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Side Panel - AI Feedback */}
        {interviewStarted && (
          <div className="absolute right-6 top-6 bottom-24 w-80 bg-gradient-to-br from-white/5 to-white/[0.02] backdrop-blur-sm rounded-2xl border border-white/10 p-6 overflow-y-auto">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Live Analytics
            </h3>

            <div className="space-y-4">
              {/* Eye Contact Metric */}
              <div className="p-4 rounded-lg bg-green-500/10 border border-white/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Eye Contact</span>
                  <span className="text-sm font-bold text-purple-400">
                    {visionActive ? 'Tracking...' : 'Waiting...'}
                  </span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div 
                    className="h-full transition-all duration-300 bg-purple-500"
                    style={{ width: visionActive ? '75%' : '0%' }}
                  />
                </div>
              </div>

              {/* Confidence Score */}
              <div className="p-4 rounded-lg bg-blue-500/10 border border-white/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">AI Confidence Score</span>
                  <span className="text-sm font-bold text-blue-400">
                    {latestConfidence !== null 
                      ? `${(latestConfidence * 100).toFixed(1)}%`
                      : predictions.length > 0
                      ? `${(predictionSummary?.mean_confidence ?? 0 * 100).toFixed(1)}%`
                      : 'Analyzing...'}
                  </span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-500"
                    style={{ 
                      width: latestConfidence !== null 
                        ? `${Math.max(0, Math.min(100, latestConfidence * 100))}%`
                        : predictions.length > 0
                        ? `${Math.max(0, Math.min(100, (predictionSummary?.mean_confidence ?? 0) * 100))}%`
                        : '0%'
                    }}
                  />
                </div>
                {predictions.length > 0 && (
                  <div className="mt-2 text-xs text-gray-500">
                    Based on {predictions.length} chunk{predictions.length !== 1 ? 's' : ''} analyzed
                  </div>
                )}
              </div>

              {/* Speaking Pace */}
              <div className="p-4 rounded-lg bg-purple-500/10 border border-white/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Speaking Pace</span>
                  <span className="text-sm font-bold text-purple-400">Normal</span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-purple-500 to-pink-400 w-[75%]" />
                </div>
              </div>

              {/* Facial Expression */}
              <div className="p-4 rounded-lg bg-green-500/10 border border-white/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Facial Expression</span>
                  <span className="text-sm font-bold text-green-400">Engaged</span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-green-500 to-emerald-400 w-[92%]" />
                </div>
              </div>

              {/* Tips */}
              <div className="mt-6 p-4 rounded-lg bg-white/5 border border-white/10">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <p className="text-sm font-medium text-yellow-400 mb-1">
                      {predictions.length > 0 ? 'AI Analysis Active' : 'Tip'}
                    </p>
                    <p className="text-sm text-gray-400">
                      {predictions.length > 0
                        ? `VideoMAE is analyzing your interview in real-time. ${predictions.length} chunk${predictions.length !== 1 ? 's' : ''} processed so far.`
                        : visionActive 
                        ? 'Maintain eye contact with the camera. AI analysis will begin after 15 seconds.'
                        : 'Start your interview to begin gaze tracking and AI analysis.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Control Bar */}
      <div className="px-6 py-6 border-t border-white/10 flex items-center justify-center gap-4">
        
        {/* Vision.py Status Indicator */}
        {visionActive && (
          <div className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 border border-purple-500/30 rounded-full">
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
            <span className="text-sm text-purple-400">Vision Window Active</span>
          </div>
        )}

        {/* Record Control */}
        <button
          onClick={() => setIsRecording(!isRecording)}
          className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
              : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
        >
          <div className={`w-6 h-6 ${isRecording ? 'rounded-sm' : 'rounded-full'} bg-white transition-all`} />
        </button>

        {/* Share Screen */}
        <button className="w-14 h-14 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center transition-all">
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
          </svg>
        </button>

        {/* Settings */}
        <button className="w-14 h-14 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center transition-all">
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>

        {/* Leave Button */}
        <button 
          onClick={handleLeave}
          className="ml-4 px-6 py-3 bg-red-500 hover:bg-red-600 text-white font-medium rounded-full transition-all flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
          Leave
        </button>
      </div>

      {/* Vision Session Results Modal */}
      {showResults && visionSessionData && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-6">
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl border border-white/20 max-w-3xl w-full max-h-[80vh] overflow-y-auto shadow-2xl">
            {/* Header */}
            <div className="p-6 border-b border-white/10">
              <h2 className="text-2xl font-bold text-white mb-2">📊 Gaze Tracking Results</h2>
              <p className="text-gray-400 text-sm">Session completed successfully</p>
            </div>

            {/* Stats Grid */}
            <div className="p-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {(() => {
                  const total = visionSessionData.log_data?.length || 0;
                  const focusedCount = visionSessionData.log_data?.filter((e: any) => 
                    !e.status.includes('Away')
                  ).length || 0;
                  const focusPercentage = total > 0 ? ((focusedCount / total) * 100).toFixed(1) : '0';
                  const lookingAwayCount = total - focusedCount;
                  
                  return (
                    <>
                      <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/10 p-4 rounded-lg border border-blue-500/30">
                        <div className="text-3xl font-bold text-blue-400">{focusPercentage}%</div>
                        <div className="text-xs text-gray-400 mt-1">Focus Score</div>
                      </div>
                      <div className="bg-gradient-to-br from-green-500/20 to-green-600/10 p-4 rounded-lg border border-green-500/30">
                        <div className="text-3xl font-bold text-green-400">{focusedCount}</div>
                        <div className="text-xs text-gray-400 mt-1">Focused</div>
                      </div>
                      <div className="bg-gradient-to-br from-red-500/20 to-red-600/10 p-4 rounded-lg border border-red-500/30">
                        <div className="text-3xl font-bold text-red-400">{lookingAwayCount}</div>
                        <div className="text-xs text-gray-400 mt-1">Looking Away</div>
                      </div>
                      <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/10 p-4 rounded-lg border border-purple-500/30">
                        <div className="text-3xl font-bold text-purple-400">{total}</div>
                        <div className="text-xs text-gray-400 mt-1">Total Events</div>
                      </div>
                    </>
                  );
                })()}
              </div>

              {/* Session Info */}
              <div className="bg-white/5 p-4 rounded-lg mb-6 border border-white/10">
                <div className="text-xs text-gray-400 space-y-2">
                  <div className="flex justify-between">
                    <span>Session ID:</span>
                    <span className="font-mono text-gray-300">{visionSessionData.session_id?.slice(0, 12)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Started:</span>
                    <span className="font-mono text-gray-300">
                      {new Date(visionSessionData.start_time).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Ended:</span>
                    <span className="font-mono text-gray-300">
                      {new Date(visionSessionData.end_time).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Duration:</span>
                    <span className="font-mono text-gray-300">
                      {Math.round((new Date(visionSessionData.end_time).getTime() - 
                                   new Date(visionSessionData.start_time).getTime()) / 1000)}s
                    </span>
                  </div>
                </div>
              </div>

              {/* Timeline */}
              <div className="bg-white/5 p-4 rounded-lg border border-white/10 max-h-64 overflow-y-auto">
                <h3 className="text-sm font-semibold text-white mb-3">Gaze Timeline (Last 20)</h3>
                <div className="space-y-2">
                  {visionSessionData.log_data?.slice(-20).reverse().map((entry: any, idx: number) => (
                    <div 
                      key={idx}
                      className={`text-xs p-2 rounded ${
                        entry.status.includes('Away')
                          ? 'bg-red-500/10 border-l-2 border-red-500'
                          : 'bg-green-500/10 border-l-2 border-green-500'
                      }`}
                    >
                      <span className="font-mono text-gray-400">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                      <span className="ml-3 text-gray-300">{entry.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="p-6 border-t border-white/10 flex gap-3">
              <button
                onClick={() => setShowResults(false)}
                className="flex-1 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all"
              >
                Close
              </button>
              <button
                onClick={() => {
                  // Download data as JSON
                  const blob = new Blob([JSON.stringify(visionSessionData, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `gaze-session-${visionSessionData.session_id.slice(0, 8)}.json`;
                  a.click();
                }}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white rounded-lg transition-all font-medium"
              >
                Download Data
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
