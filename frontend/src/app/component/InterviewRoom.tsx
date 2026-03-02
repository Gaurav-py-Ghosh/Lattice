'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useVisionSession } from '../hooks/useVisionSession';
import { useChunkedRecorder } from '../hooks/useChunkedRecorder';
import CalibrationFlow from './CalibrationFlow';

// ---------------------------------------------------------------------------
// Static question bank — at module scope to avoid re-allocation on every render
// ---------------------------------------------------------------------------
const INTERVIEW_QUESTIONS = [
  'Tell me about yourself and your background.',
  'Describe a time when you had to work under pressure. How did you handle it?',
  'What is your greatest professional achievement?',
  'Tell me about a challenge you faced at work and how you overcame it.',
  'Where do you see yourself in five years?',
  'Why are you interested in this role?',
  'Describe a situation where you had to work with a difficult team member.',
  'What are your greatest strengths and how have they helped you professionally?',
  'Tell me about a time you failed and what you learned from it.',
  'Do you have any questions for us?',
];

export default function InterviewRoom() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [recordingTime, setRecordingTime] = useState(0);
  const [showCalibration, setShowCalibration] = useState(true);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [visionSessionData, setVisionSessionData] = useState<any>(null);
  const [showResults, setShowResults] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [permissionStatus, setPermissionStatus] = useState<'pending' | 'granted' | 'denied'>('pending');

  const [questionIndex, setQuestionIndex] = useState(0);
  const questionTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Vision.py session hook
  const {
    isConnected: visionConnected,
    sessionData,
    latestConfidence,
    chunkResults,
    latestVoiceScore,
    latestFacialScore,
    pendingChunks,
    chunkErrors,
    dismissChunkError,
    processChunk,
    error: visionError,
  } = useVisionSession();

  // Chunked recorder — 15-s chunks auto-uploaded to vision_server
  const {
    stream: cameraStream,
    isRecording: isChunkRecording,
    chunkCount,
    pendingUploads,
    permissionGranted,
    permissionError,
    requestPermissions,
    start: startRecorder,
    stop: stopRecorder,
    releaseStream,
  } = useChunkedRecorder({
    onChunkReady: (videoPath, chunkId, chunkIndex) => {
      console.log(`📦 Chunk ${chunkIndex} ready: ${videoPath}`);
      processChunk(videoPath, chunkId, chunkIndex);
    },
    onError: (msg) => console.error('[Recorder]', msg),
  });

  // Request camera + microphone on mount and feed stream into video element
  useEffect(() => {
    requestPermissions().then((s) => {
      if (s) {
        setPermissionStatus('granted');
        if (videoRef.current) {
          videoRef.current.srcObject = s;
        }
      } else {
        setPermissionStatus('denied');
      }
    });

    return () => {
      releaseStream();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
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

  // Recording timer tied to chunked recorder
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isChunkRecording) {
      timer = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else {
      setRecordingTime(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isChunkRecording]);

  // Show results when chunks have been analyzed and recording stops
  useEffect(() => {
    if (chunkResults.length > 0 && !isChunkRecording && interviewStarted) {
      console.log('📊 Got chunk results:', chunkResults.length);
      setVisionSessionData({ chunkResults, sessionData });
    }
  }, [chunkResults, isChunkRecording, interviewStarted, sessionData]);

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

  // Auto-advance questions every 90 seconds while recording
  useEffect(() => {
    if (interviewStarted && isChunkRecording) {
      questionTimerRef.current = setInterval(() => {
        setQuestionIndex((prev) =>
          prev < INTERVIEW_QUESTIONS.length - 1 ? prev + 1 : prev
        );
      }, 90_000);
    }
    return () => {
      if (questionTimerRef.current) {
        clearInterval(questionTimerRef.current);
        questionTimerRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interviewStarted, isChunkRecording]);

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

    // Start 15-s chunk recording now that permissions are already granted
    if (permissionGranted && cameraStream) {
      console.log('🎥 Starting chunked recording (15-s chunks) ...');
      startRecorder(cameraStream);
    } else {
      console.warn('Camera stream not ready — requesting permissions again');
      requestPermissions().then((s) => {
        if (s) {
          startRecorder(s);
          if (videoRef.current) videoRef.current.srcObject = s;
        }
      });
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleLeave = async () => {
    // Exit fullscreen first
    await exitFullscreen();

    // Stop chunked recording (flushes final chunk)
    if (isChunkRecording) {
      console.log('⏹️ Stopping chunked recording ...');
      stopRecorder();
    }

    // Always show results modal if interview was started
    if (interviewStarted) {
      setVisionSessionData({ chunkResults, sessionData });
      setShowResults(true);
    } else {
      setTimeout(() => {
        releaseStream();
        router.push('/');
      }, 500);
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
          {isChunkRecording && (
            <div className="flex items-center gap-2 px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-full">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm text-red-400 font-mono">● Chunk {chunkCount} | {formatTime(recordingTime)}</span>
            </div>
          )}
          {pendingChunks > 0 && (
            <div className="flex items-center gap-2 px-3 py-1 bg-yellow-500/20 border border-yellow-500/30 rounded-full">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              <span className="text-sm text-yellow-400 font-mono">Analyzing {pendingChunks} chunk{pendingChunks !== 1 ? 's' : ''}…</span>
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
          {/* Permission Status */}
          {permissionStatus === 'denied' && (
            <div className="px-3 py-1.5 bg-red-500/20 border border-red-500/30 rounded-lg text-xs text-red-400">
              Camera/Mic Denied
            </div>
          )}


          
          <button 
            onClick={() => {
              if (confirm('End the interview and see your results?')) {
                handleLeave();
              }
            }}
            className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
          >
            End Interview
          </button>
        </div>
      </div>

      {/* Chunk Error Toasts */}
      {chunkErrors.length > 0 && (
        <div className="fixed bottom-24 right-4 z-50 flex flex-col gap-2 max-w-sm">
          {chunkErrors.map((err, i) => (
            <div key={i} className="flex items-start gap-3 px-4 py-3 bg-red-900/80 border border-red-500/40 rounded-xl backdrop-blur-sm text-sm text-red-200 shadow-lg">
              <svg className="w-4 h-4 mt-0.5 flex-shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="flex-1 font-mono text-xs">{err}</span>
              <button onClick={() => dismissChunkError(i)} className="text-red-400 hover:text-white">✕</button>
            </div>
          ))}
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center p-6 relative">
        {/* Background Effects */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] opacity-50" />
        </div>

        {/* Video Container */}
        <div className="relative w-full max-w-5xl aspect-video">
          {/* Live camera feed (browser-side) */}
          <div className="relative w-full h-full bg-black rounded-2xl overflow-hidden border-2 border-white/10 shadow-2xl">

            {/* Live video preview */}
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className={`w-full h-full object-cover ${
                permissionStatus === 'granted' ? 'block' : 'hidden'
              }`}
            />

            {/* Permission / waiting state */}
            {permissionStatus !== 'granted' && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center max-w-md p-8">
                  <div className="w-24 h-24 mx-auto mb-6 bg-white/5 rounded-full flex items-center justify-center border-4 border-white/10">
                    <svg className="w-12 h-12 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  {permissionStatus === 'pending' ? (
                    <>
                      <h3 className="text-xl font-bold text-white mb-2">Requesting Camera & Mic</h3>
                      <p className="text-gray-400 text-sm">Please allow access in your browser prompt.</p>
                    </>
                  ) : (
                    <>
                      <h3 className="text-xl font-bold text-red-400 mb-2">Permission Denied</h3>
                      <p className="text-gray-400 text-sm">{permissionError ?? 'Enable camera and microphone access, then refresh.'}</p>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Recording badge */}
            {isChunkRecording && (
              <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 bg-red-600/80 backdrop-blur-sm rounded-full">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                <span className="text-xs font-semibold text-white">REC</span>
              </div>
            )}

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
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm text-purple-400">
                      Question {questionIndex + 1} of {INTERVIEW_QUESTIONS.length}
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setQuestionIndex((p) => Math.max(0, p - 1))}
                        disabled={questionIndex === 0}
                        className="px-2 py-1 text-xs text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                      >
                        ← Prev
                      </button>
                      <button
                        onClick={() => setQuestionIndex((p) => Math.min(INTERVIEW_QUESTIONS.length - 1, p + 1))}
                        disabled={questionIndex === INTERVIEW_QUESTIONS.length - 1}
                        className="px-2 py-1 text-xs text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
                      >
                        Next →
                      </button>
                    </div>
                  </div>
                  {/* Progress bar */}
                  <div className="h-0.5 bg-white/10 rounded-full mb-3 overflow-hidden">
                    <div
                      className="h-full bg-purple-500 transition-all duration-500"
                      style={{ width: `${((questionIndex + 1) / INTERVIEW_QUESTIONS.length) * 100}%` }}
                    />
                  </div>
                  <p className="text-lg text-white">
                    {INTERVIEW_QUESTIONS[questionIndex]}
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
                    {chunkResults.length > 0
                      ? (() => {
                          const total = chunkResults.reduce((s, c) => s + c.gaze_data.length, 0);
                          const focused = chunkResults.reduce(
                            (s, c) => s + c.gaze_data.filter((e) => !e.status.includes('Away')).length, 0
                          );
                          return total > 0 ? `${((focused / total) * 100).toFixed(0)}%` : 'Tracking…';
                        })()
                      : 'Tracking…'}
                  </span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all duration-300 bg-purple-500"
                    style={{
                      width: chunkResults.length > 0
                        ? (() => {
                            const total = chunkResults.reduce((s, c) => s + c.gaze_data.length, 0);
                            const focused = chunkResults.reduce(
                              (s, c) => s + c.gaze_data.filter((e) => !e.status.includes('Away')).length, 0
                            );
                            return total > 0 ? `${((focused / total) * 100).toFixed(0)}%` : '0%';
                          })()
                        : '0%',
                    }}
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
                      : chunkResults.length > 0
                      ? (() => {
                          const allPreds = chunkResults.flatMap((c) => c.predictions);
                          const avg = allPreds.length > 0
                            ? allPreds.reduce((s, p) => s + p.confidence, 0) / allPreds.length
                            : null;
                          return avg !== null ? `${(avg * 100).toFixed(1)}%` : 'Analyzing…';
                        })()
                      : 'Analyzing…'}
                  </span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-500"
                    style={{
                      width: latestConfidence !== null
                        ? `${Math.max(0, Math.min(100, latestConfidence * 100))}%`
                        : '0%'
                    }}
                  />
                </div>
                {chunkResults.length > 0 && (
                  <div className="mt-2 text-xs text-gray-500">
                    Based on {chunkResults.length} chunk{chunkResults.length !== 1 ? 's' : ''} analyzed
                  </div>
                )}
              </div>

              {/* Voice Analysis Score */}
              <div className="p-4 rounded-lg bg-emerald-500/10 border border-white/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Voice Skills</span>
                  <span className="text-sm font-bold text-emerald-400">
                    {latestVoiceScore !== null
                      ? `${(latestVoiceScore * 100).toFixed(1)}%`
                      : pendingChunks > 0
                      ? 'Analyzing…'
                      : 'Waiting…'}
                  </span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-500"
                    style={{ width: latestVoiceScore !== null ? `${Math.min(100, latestVoiceScore * 100).toFixed(0)}%` : '0%' }}
                  />
                </div>
              </div>

              {/* Facial Expression */}
              <div className="p-4 rounded-lg bg-green-500/10 border border-white/10">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Facial Expression</span>
                  <span className="text-sm font-bold text-green-400">
                    {latestFacialScore !== null
                      ? `${(latestFacialScore * 100).toFixed(1)}%`
                      : pendingChunks > 0
                      ? 'Analyzing…'
                      : 'Waiting…'}
                  </span>
                </div>
                <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-emerald-400 transition-all duration-500"
                    style={{ width: latestFacialScore !== null ? `${Math.min(100, latestFacialScore * 100).toFixed(0)}%` : '0%' }}
                  />
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
                      {chunkResults.length > 0 ? 'AI Analysis Active' : 'Tip'}
                    </p>
                    <p className="text-sm text-gray-400">
                      {chunkResults.length > 0
                        ? `${chunkResults.length} chunk${chunkResults.length !== 1 ? 's' : ''} analyzed.${pendingUploads > 0 ? ` ${pendingUploads} uploading…` : ''}${pendingChunks > 0 ? ` ${pendingChunks} processing…` : ''}`
                        : isChunkRecording
                        ? 'Recording in progress. First AI scores appear after 15 seconds.'
                        : 'Complete calibration to begin recording and AI analysis.'}
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
        
        {/* Recording / Processing Indicator */}
        {isChunkRecording && (
          <div className="flex items-center gap-2 px-4 py-2 bg-red-500/20 border border-red-500/30 rounded-full">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-sm text-red-400">Recording — Chunk {chunkCount}</span>
          </div>
        )}

        {/* Record Control */}
        <button
          onClick={() => {
            if (isChunkRecording) {
              stopRecorder();
            } else if (cameraStream) {
              setQuestionIndex(0);
              startRecorder(cameraStream);
            }
          }}
          className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
            isChunkRecording
              ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
              : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
        >
          <div className={`w-6 h-6 ${isChunkRecording ? 'rounded-sm' : 'rounded-full'} bg-white transition-all`} />
        </button>

        {/* Share Screen */}
        {/* Settings */}

        {/* Leave Button */}
        <button 
          onClick={() => { if (confirm('End the interview and see your results?')) handleLeave(); }}
          className="ml-4 px-6 py-3 bg-red-500 hover:bg-red-600 text-white font-medium rounded-full transition-all flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
          Leave
        </button>
      </div>

      {/* Results Modal */}
      {showResults && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-6">
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl border border-white/20 max-w-3xl w-full max-h-[80vh] overflow-y-auto shadow-2xl">
            {/* Header */}
            <div className="p-6 border-b border-white/10">
              <h2 className="text-2xl font-bold text-white mb-2">📊 Interview Results</h2>
              <p className="text-gray-400 text-sm">
                {chunkResults.length} chunk{chunkResults.length !== 1 ? 's' : ''} analyzed
                {pendingUploads > 0 && (
                  <span className="ml-2 text-blue-400 animate-pulse">
                    · {pendingUploads} uploading…
                  </span>
                )}
                {pendingChunks > 0 && (
                  <span className="ml-2 text-yellow-400 animate-pulse">
                    · {pendingChunks} still processing…
                  </span>
                )}
              </p>
            </div>

            <div className="p-6">
              {/* Aggregated Stats */}
              {(() => {
                const allGaze = chunkResults.flatMap((c) => c.gaze_data);
                const total = allGaze.length;
                const focused = allGaze.filter((e) => !e.status.includes('Away')).length;
                const focusPct = total > 0 ? ((focused / total) * 100).toFixed(1) : '0';
                const avgVoice =
                  chunkResults.filter((c) => c.voice_analysis?.score != null).length > 0
                    ? (
                        chunkResults
                          .filter((c) => c.voice_analysis?.score != null)
                          .reduce((s, c) => s + c.voice_analysis!.score!, 0) /
                        chunkResults.filter((c) => c.voice_analysis?.score != null).length
                      ).toFixed(3)
                    : null;
                const allPreds = chunkResults.flatMap((c) => c.predictions);
                const avgConf =
                  allPreds.length > 0
                    ? (allPreds.reduce((s, p) => s + p.confidence, 0) / allPreds.length).toFixed(3)
                    : null;
                const avgFacial =
                  chunkResults.filter((c) => c.facial_analysis?.score != null).length > 0
                    ? (
                        chunkResults
                          .filter((c) => c.facial_analysis?.score != null)
                          .reduce((s, c) => s + c.facial_analysis!.score!, 0) /
                        chunkResults.filter((c) => c.facial_analysis?.score != null).length
                      ).toFixed(3)
                    : null;

                return (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/10 p-4 rounded-lg border border-blue-500/30">
                      <div className="text-3xl font-bold text-blue-400">{focusPct}%</div>
                      <div className="text-xs text-gray-400 mt-1">Focus Score</div>
                    </div>
                    <div className="bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 p-4 rounded-lg border border-emerald-500/30">
                      <div className="text-3xl font-bold text-emerald-400">
                        {avgVoice ? `${(parseFloat(avgVoice) * 100).toFixed(0)}%` : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Voice Skills</div>
                    </div>
                    <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/10 p-4 rounded-lg border border-purple-500/30">
                      <div className="text-3xl font-bold text-purple-400">
                        {avgConf ? `${(parseFloat(avgConf) * 100).toFixed(0)}%` : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Avg Confidence</div>
                    </div>
                    <div className="bg-gradient-to-br from-green-500/20 to-green-600/10 p-4 rounded-lg border border-green-500/30">
                      <div className="text-3xl font-bold text-green-400">
                        {avgFacial ? `${(parseFloat(avgFacial) * 100).toFixed(0)}%` : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Facial Expression</div>
                    </div>
                  </div>
                );
              })()}

              {/* Per-chunk breakdown */}
              <div className="bg-white/5 p-4 rounded-lg border border-white/10 max-h-64 overflow-y-auto">
                <h3 className="text-sm font-semibold text-white mb-3">Per-Chunk Breakdown</h3>
                <div className="space-y-2">
                  {chunkResults.map((chunk, idx) => (
                    <div key={chunk.chunkId} className="text-xs p-3 rounded bg-white/5 border border-white/10">
                      <div className="flex justify-between items-center">
                        <span className="font-mono text-gray-400">Chunk {idx + 1}</span>
                        <div className="flex gap-3">
                          {chunk.voice_analysis?.score != null && (
                            <span className="text-emerald-400">
                              Voice: {(chunk.voice_analysis.score * 100).toFixed(0)}%
                            </span>
                          )}
                          {chunk.predictions.length > 0 && (
                            <span className="text-purple-400">
                              Conf: {(chunk.predictions[chunk.predictions.length - 1].confidence * 100).toFixed(0)}%
                            </span>
                          )}
                          {chunk.facial_analysis?.score != null && (
                            <span className="text-green-400">
                              Facial: {(chunk.facial_analysis.score * 100).toFixed(0)}%
                            </span>
                          )}
                          <span className="text-gray-400">
                            Gaze: {chunk.gaze_data.filter((e) => !e.status.includes('Away')).length}/{chunk.gaze_data.length}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                  {chunkResults.length === 0 && (
                    <p className="text-gray-500 text-xs text-center py-4">
                      {pendingChunks > 0 ? 'Waiting for first chunk to complete…' : 'No chunks analyzed yet.'}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="p-6 border-t border-white/10 flex gap-3">
              <button
                onClick={() => {
                  if (pendingChunks > 0) {
                    if (!confirm(`${pendingChunks} chunk${pendingChunks !== 1 ? 's are' : ' is'} still processing. Results will be lost. Leave anyway?`)) return;
                  }
                  releaseStream();
                  router.push('/');
                }}
                className="flex-1 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all"
              >
                Close & Go Home
              </button>
              <button
                disabled={pendingChunks > 0 || pendingUploads > 0}
                onClick={() => {
                  // Aggregate inference_summary across all chunks
                  const allConf = chunkResults.flatMap((c) => c.predictions.map((p) => p.confidence));
                  const overall_inference_summary = allConf.length > 0
                    ? {
                        count: allConf.length,
                        mean_confidence: parseFloat((allConf.reduce((s, v) => s + v, 0) / allConf.length).toFixed(4)),
                        min_confidence:  parseFloat(Math.min(...allConf).toFixed(4)),
                        max_confidence:  parseFloat(Math.max(...allConf).toFixed(4)),
                      }
                    : null;
                  const blob = new Blob(
                    [JSON.stringify({ chunkResults, overall_inference_summary, sessionData }, null, 2)],
                    { type: 'application/json' }
                  );
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `interview-results-${Date.now()}.json`;
                  a.click();
                }}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white rounded-lg transition-all font-medium disabled:opacity-40 disabled:cursor-not-allowed disabled:from-purple-500/50 disabled:to-pink-500/50"
                title={pendingChunks > 0 || pendingUploads > 0 ? `Waiting for ${pendingUploads} upload${pendingUploads !== 1 ? 's' : ''} + ${pendingChunks} chunk${pendingChunks !== 1 ? 's' : ''} to finish…` : 'Download all results'}
              >
                {pendingUploads > 0
                  ? `Uploading ${pendingUploads} chunk${pendingUploads !== 1 ? 's' : ''}…`
                  : pendingChunks > 0
                  ? `Waiting for ${pendingChunks} chunk${pendingChunks !== 1 ? 's' : ''}…`
                  : 'Download Data'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
