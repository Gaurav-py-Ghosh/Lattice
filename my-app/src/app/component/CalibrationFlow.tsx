'use client';

import { useState, useEffect } from 'react';

interface CalibrationFlowProps {
  onComplete: () => void;
  onCalibrate: () => void;
  calibrated: boolean;
  screenCalibrated: boolean;
}

export default function CalibrationFlow({ 
  onComplete, 
  onCalibrate, 
  calibrated, 
  screenCalibrated 
}: CalibrationFlowProps) {
  const [step, setStep] = useState<'intro' | 'look-center' | 'calibrating' | 'ready'>('intro');
  const [countdown, setCountdown] = useState(3);

  useEffect(() => {
    if (step === 'calibrating' && countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    } else if (step === 'calibrating' && countdown === 0) {
      // Trigger calibration
      onCalibrate();
      setTimeout(() => {
        setStep('ready');
      }, 1000);
    }
  }, [step, countdown, onCalibrate]);

  const handleNext = () => {
    if (step === 'intro') {
      setStep('look-center');
    } else if (step === 'look-center') {
      setStep('calibrating');
      setCountdown(3);
    }
  };

  const handleStartInterview = () => {
    onComplete();
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-md z-50 flex items-center justify-center">
      <div className="max-w-2xl w-full px-6">
        <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl border border-white/20 p-8 shadow-2xl">
          
          {/* Intro Step */}
          {step === 'intro' && (
            <div className="text-center space-y-6 animate-fade-in">
              <div className="w-20 h-20 mx-auto bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
                <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              
              <h2 className="text-3xl font-bold text-white">
                Eye Tracking Calibration
              </h2>
              
              <p className="text-lg text-gray-300">
                We'll calibrate the eye tracking system to ensure accurate monitoring during your interview.
              </p>

              <div className="bg-white/5 border border-white/10 rounded-xl p-6 text-left space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-purple-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-purple-400 text-sm font-bold">1</span>
                  </div>
                  <p className="text-gray-300">Position yourself comfortably in front of the camera</p>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-purple-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-purple-400 text-sm font-bold">2</span>
                  </div>
                  <p className="text-gray-300">Look directly at the center of the screen</p>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-purple-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-purple-400 text-sm font-bold">3</span>
                  </div>
                  <p className="text-gray-300">Hold your gaze steady for 3 seconds</p>
                </div>
              </div>

              <button
                onClick={handleNext}
                className="w-full py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-xl transition-all shadow-lg hover:shadow-xl"
              >
                Begin Calibration
              </button>
            </div>
          )}

          {/* Look at Center Step */}
          {step === 'look-center' && (
            <div className="text-center space-y-8 animate-fade-in">
              <h2 className="text-2xl font-bold text-white">
                Look at the Center Target
              </h2>
              
              <div className="relative h-64 flex items-center justify-center">
                {/* Crosshair target */}
                <div className="relative">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-16 h-16 border-4 border-purple-500 rounded-full animate-ping opacity-20"></div>
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-12 h-12 border-4 border-purple-500 rounded-full"></div>
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-4 h-4 bg-purple-500 rounded-full shadow-lg shadow-purple-500/50"></div>
                  </div>
                  
                  {/* Crosshair lines */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-24 h-0.5 bg-purple-500/50"></div>
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-0.5 h-24 bg-purple-500/50"></div>
                  </div>
                </div>
              </div>

              <p className="text-gray-300">
                Focus on the center dot and keep your head steady
              </p>

              <div className="flex gap-3">
                <button
                  onClick={() => setStep('intro')}
                  className="flex-1 py-3 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl transition-all"
                >
                  Back
                </button>
                <button
                  onClick={handleNext}
                  className="flex-1 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-xl transition-all"
                >
                  I'm Ready
                </button>
              </div>
            </div>
          )}

          {/* Calibrating Step */}
          {step === 'calibrating' && (
            <div className="text-center space-y-8 animate-fade-in">
              <h2 className="text-2xl font-bold text-white">
                Calibrating...
              </h2>
              
              <div className="relative h-64 flex items-center justify-center">
                <div className="text-8xl font-bold text-purple-500 animate-pulse">
                  {countdown > 0 ? countdown : '✓'}
                </div>
              </div>

              <p className="text-gray-300">
                {countdown > 0 ? 'Hold steady...' : 'Calibration complete!'}
              </p>

              <div className="flex items-center justify-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-75"></div>
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-150"></div>
              </div>
            </div>
          )}

          {/* Ready Step */}
          {step === 'ready' && (
            <div className="text-center space-y-6 animate-fade-in">
              <div className="w-20 h-20 mx-auto bg-green-500/20 rounded-2xl flex items-center justify-center">
                <svg className="w-10 h-10 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              
              <h2 className="text-3xl font-bold text-white">
                All Set!
              </h2>
              
              <p className="text-lg text-gray-300">
                Eye tracking is calibrated and ready. You can now start your interview.
              </p>

              <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Auto Calibration</span>
                  <span className="flex items-center gap-2 text-green-400 font-medium">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Complete
                  </span>
                </div>
                {screenCalibrated && (
                  <div className="flex items-center justify-between mt-2 pt-2 border-t border-green-500/10">
                    <span className="text-gray-300">Screen Calibration</span>
                    <span className="flex items-center gap-2 text-green-400 font-medium">
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Complete
                    </span>
                  </div>
                )}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={onCalibrate}
                  className="flex-1 py-3 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl transition-all"
                >
                  Recalibrate
                </button>
                <button
                  onClick={handleStartInterview}
                  className="flex-1 py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-xl transition-all shadow-lg hover:shadow-xl"
                >
                  Start Interview
                </button>
              </div>
            </div>
          )}

        </div>

        {/* Status Info */}
        <div className="mt-4 text-center text-sm text-gray-400">
          {!calibrated && <p>Waiting for face detection...</p>}
          {calibrated && !screenCalibrated && <p>Auto-calibration detected</p>}
          {screenCalibrated && <p>Screen calibration applied</p>}
        </div>
      </div>
    </div>
  );
}
