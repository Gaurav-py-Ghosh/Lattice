/**
 * Example: Interview Page with Vision Session Control
 * 
 * This example shows how to integrate the VisionSessionControl component
 * into your interview page for gaze tracking
 */

'use client';

import React, { useState } from 'react';
import VisionSessionControl from '../component/VisionSessionControl';

export default function InterviewWithVisionTracking() {
  const [sessionResults, setSessionResults] = useState<any>(null);
  const [interviewStarted, setInterviewStarted] = useState(false);

  const handleSessionEnd = (sessionData: any) => {
    console.log('📊 Gaze tracking session ended:', sessionData);
    setSessionResults(sessionData);
    
    // You can process the data here:
    // - Save to database
    // - Calculate metrics
    // - Show feedback to user
  };

  const handleStartInterview = () => {
    setInterviewStarted(true);
    // Your interview logic here
  };

  const handleEndInterview = () => {
    setInterviewStarted(false);
    // Your interview logic here
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Mock Interview Session
          </h1>
          <p className="text-gray-600">
            Practice with AI-powered gaze tracking and feedback
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Main Interview Area */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl p-6">
              
              {/* Video/Interview Area */}
              <div className="bg-gray-900 rounded-lg aspect-video flex items-center justify-center mb-6">
                <div className="text-center text-white">
                  <div className="text-6xl mb-4">🎥</div>
                  <p className="text-lg">Your camera feed will appear here</p>
                  <p className="text-sm text-gray-400 mt-2">
                    {interviewStarted ? 'Interview in progress...' : 'Click "Start Interview" to begin'}
                  </p>
                </div>
              </div>

              {/* Interview Controls */}
              <div className="flex gap-4">
                {!interviewStarted ? (
                  <button
                    onClick={handleStartInterview}
                    className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 transition-all shadow-lg"
                  >
                    ▶️ Start Interview
                  </button>
                ) : (
                  <button
                    onClick={handleEndInterview}
                    className="flex-1 bg-gradient-to-r from-red-500 to-rose-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-red-600 hover:to-rose-700 transition-all shadow-lg"
                  >
                    ⏹️ End Interview
                  </button>
                )}
              </div>

              {/* Current Question */}
              {interviewStarted && (
                <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                  <p className="text-sm text-blue-600 mb-1">Current Question:</p>
                  <p className="text-gray-800 font-medium">
                    "Tell me about a time you faced a challenging problem and how you solved it."
                  </p>
                </div>
              )}

            </div>
          </div>

          {/* Right Sidebar - Vision Tracking */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">
                👁️ Gaze Tracking
              </h2>
              
              {/* Vision Session Control Component */}
              <VisionSessionControl 
                onSessionEnd={handleSessionEnd}
                autoStart={false}  // Set to true to auto-start with page load
              />
              
              {/* Instructions */}
              <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-xs text-yellow-800 font-semibold mb-2">
                  📝 Instructions:
                </p>
                <ul className="text-xs text-yellow-700 space-y-1">
                  <li>1. Click "Start Gaze Tracking"</li>
                  <li>2. Calibrate when window appears (press C)</li>
                  <li>3. Start your interview</li>
                  <li>4. Stop tracking when done</li>
                  <li>5. Review your focus metrics</li>
                </ul>
              </div>

              {/* Quick Stats Display */}
              {sessionResults && (
                <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-sm font-semibold text-green-800 mb-2">
                    ✅ Last Session Summary
                  </p>
                  <div className="text-xs text-green-700 space-y-1">
                    <div>Focus Score: <span className="font-bold">{
                      ((sessionResults.log_data.filter((e: any) => 
                        !e.status.includes('Away')
                      ).length / sessionResults.log_data.length) * 100).toFixed(1)
                    }%</span></div>
                    <div>Duration: {
                      Math.round((new Date(sessionResults.end_time).getTime() - 
                                new Date(sessionResults.start_time).getTime()) / 1000)
                    }s</div>
                  </div>
                </div>
              )}

            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
