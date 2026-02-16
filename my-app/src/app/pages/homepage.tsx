'use client';

import React, { useEffect, useRef, useState } from 'react';

export default function HomePage() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('.fade-in-section').forEach((el) => {
      observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const features = [
    {
      title: 'Real-time Behavior Analysis',
      description: 'Advanced ML models analyze facial expressions, body language, and micro-gestures to provide instant feedback on candidate confidence and engagement.',
      metric: '98.7%',
      metricLabel: 'Accuracy',
      delay: '0s'
    },
    {
      title: 'Adaptive Question Engine',
      description: 'Dynamic interview flow that adjusts difficulty and topics based on real-time performance indicators and behavioral cues.',
      metric: '3.2x',
      metricLabel: 'Faster Assessment',
      delay: '0.1s'
    },
    {
      title: 'Voice & Speech Analytics',
      description: 'Comprehensive analysis of speaking patterns, pace, clarity, and vocal confidence with actionable improvement recommendations.',
      metric: '15+',
      metricLabel: 'Voice Metrics',
      delay: '0.2s'
    },
    {
      title: 'AR-Enhanced Feedback',
      description: 'Immersive augmented reality overlays that visualize performance metrics and provide spatial awareness training for interviews.',
      metric: '4K',
      metricLabel: 'Video Quality',
      delay: '0.3s'
    }
  ];

  const stats = [
    { value: '250K+', label: 'Interviews Conducted' },
    { value: '94%', label: 'Success Rate' },
    { value: '2.8M', label: 'Data Points Analyzed' },
    { value: '<100ms', label: 'Response Time' }
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-hidden">
      {/* Animated Grid Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_50%,black,transparent)]" />
        <div 
          className="absolute inset-0 opacity-30"
          style={{
            background: `radial-gradient(600px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(139, 92, 246, 0.08), transparent 40%)`
          }}
        />
      </div>

      {/* Navigation */}
      <nav className="relative z-50 px-6 py-6 flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center font-bold text-sm">
            AI
          </div>
          <span className="text-xl font-bold tracking-tight">InterviewAR</span>
        </div>
        
        <div className="hidden md:flex items-center gap-8 text-sm">
          <a href="#product" className="text-gray-400 hover:text-white transition-colors">Product</a>
          <a href="#features" className="text-gray-400 hover:text-white transition-colors">Features</a>
          <a href="#technology" className="text-gray-400 hover:text-white transition-colors">Technology</a>
          <a href="#pricing" className="text-gray-400 hover:text-white transition-colors">Pricing</a>
        </div>

        <div className="flex items-center gap-4">
          <button className="text-sm text-gray-400 hover:text-white transition-colors">
            Log in
          </button>
          <button className="px-4 py-2 bg-white text-black text-sm font-medium rounded-lg hover:bg-gray-100 transition-all">
            Get started
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 px-6 pt-20 pb-32 max-w-7xl mx-auto" ref={heroRef}>
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-500/10 border border-purple-500/20 rounded-full text-sm text-purple-300 mb-8 animate-fade-in">
            <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-pulse" />
            AI-Powered Interview Intelligence
          </div>
          
          <h1 className="text-6xl md:text-8xl font-bold leading-[0.9] mb-8 animate-fade-in-up tracking-tight">
            Bring
            <br />
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
              everyone
            </span>
            <br />
            together
            <br />
            <span className="text-gray-400">with data</span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mb-12 leading-relaxed animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
            Go end-to-end from quick behavioral queries to deep-dive analyses to beautiful interactive AR interviews – all in one collaborative, AI-powered workspace.
          </p>

          <div className="flex flex-wrap gap-4 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
            <button className="px-6 py-3 bg-white text-black font-medium rounded-lg hover:bg-gray-100 transition-all hover:scale-105">
              Get started for free
            </button>
            <button className="px-6 py-3 bg-transparent border border-gray-700 text-white font-medium rounded-lg hover:border-gray-500 transition-all">
              Request a demo
            </button>
          </div>
        </div>

        {/* Floating Data Visualization */}
        <div className="absolute right-0 top-20 w-[500px] h-[400px] hidden xl:block">
          <div className="relative w-full h-full animate-float">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl backdrop-blur-sm border border-white/10 p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-xs text-gray-400 font-mono">LIVE ANALYSIS</span>
              </div>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">Confidence Score</span>
                  <span className="text-lg font-bold text-green-400">87%</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-green-500 to-emerald-400 w-[87%] animate-pulse-slow" />
                </div>

                <div className="flex justify-between items-center mt-6">
                  <span className="text-sm text-gray-300">Facial Expression</span>
                  <span className="text-lg font-bold text-blue-400">Engaged</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 w-[92%] animate-pulse-slow" style={{ animationDelay: '0.5s' }} />
                </div>

                <div className="flex justify-between items-center mt-6">
                  <span className="text-sm text-gray-300">Speech Clarity</span>
                  <span className="text-lg font-bold text-purple-400">Excellent</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-purple-500 to-pink-400 w-[94%] animate-pulse-slow" style={{ animationDelay: '1s' }} />
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-white/10">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-white">2.3s</div>
                    <div className="text-xs text-gray-400">Avg Response</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">156</div>
                    <div className="text-xs text-gray-400">WPM</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">12</div>
                    <div className="text-xs text-gray-400">Questions</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 px-6 pb-20 max-w-7xl mx-auto fade-in-section">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div 
              key={index} 
              className="text-center animate-fade-in-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent mb-2">
                {stat.value}
              </div>
              <div className="text-sm text-gray-500">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 px-6 py-32 max-w-7xl mx-auto fade-in-section">
        <div className="mb-16">
          <h2 className="text-5xl font-bold mb-4">
            Next-generation
            <br />
            <span className="text-gray-600">interview intelligence</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative p-8 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-2xl border border-white/10 hover:border-purple-500/50 transition-all duration-500 animate-fade-in-up"
              style={{ animationDelay: feature.delay }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/0 to-pink-500/0 group-hover:from-purple-500/10 group-hover:to-pink-500/5 rounded-2xl transition-all duration-500" />
              
              <div className="relative">
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-2xl font-bold">{feature.title}</h3>
                  <div className="text-right">
                    <div className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      {feature.metric}
                    </div>
                    <div className="text-xs text-gray-500">{feature.metricLabel}</div>
                  </div>
                </div>
                
                <p className="text-gray-400 leading-relaxed">
                  {feature.description}
                </p>

                <div className="mt-6 flex items-center gap-2 text-sm text-purple-400 group-hover:text-purple-300 transition-colors">
                  Learn more
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Technology Section */}
      <section className="relative z-10 px-6 py-32 max-w-7xl mx-auto fade-in-section">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-sm text-blue-300 mb-6">
              Powered by Advanced ML
            </div>
            <h2 className="text-5xl font-bold mb-6">
              Ranking-based
              <br />
              <span className="text-gray-600">machine learning</span>
            </h2>
            <p className="text-xl text-gray-400 mb-8 leading-relaxed">
              Our proprietary ranking algorithms analyze hundreds of behavioral micro-signals simultaneously, providing nuanced assessment that goes far beyond traditional keyword matching or simple sentiment analysis.
            </p>
            <ul className="space-y-4">
              {[
                'Multi-modal analysis (video, audio, text)',
                'Temporal pattern recognition',
                'Context-aware feedback generation',
                'Bias-free candidate evaluation'
              ].map((item, index) => (
                <li key={index} className="flex items-start gap-3">
                  <svg className="w-6 h-6 text-purple-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-gray-300">{item}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="relative h-[500px]">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl backdrop-blur-sm border border-white/10 p-8 overflow-hidden">
              <div className="space-y-3">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div 
                    key={i}
                    className="flex items-center gap-4 animate-slide-in"
                    style={{ 
                      animationDelay: `${i * 0.1}s`,
                      opacity: 1 - (i * 0.08)
                    }}
                  >
                    <div className="w-2 h-2 bg-purple-400 rounded-full" />
                    <div className="flex-1 h-8 bg-gradient-to-r from-purple-500/40 to-transparent rounded" 
                         style={{ width: `${Math.random() * 40 + 60}%` }} 
                    />
                    <div className="text-xs text-gray-500 font-mono">{(Math.random() * 0.3 + 0.7).toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 px-6 py-32 max-w-7xl mx-auto fade-in-section">
        <div className="relative p-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-3xl backdrop-blur-sm border border-white/10 overflow-hidden">
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS1vcGFjaXR5PSIwLjA1IiBzdHJva2Utd2lkdGg9IjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')] opacity-50" />
          
          <div className="relative text-center max-w-3xl mx-auto">
            <h2 className="text-5xl font-bold mb-6">
              Ready to transform your
              <br />
              <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                interview process?
              </span>
            </h2>
            <p className="text-xl text-gray-300 mb-10">
              Join thousands of organizations using AI-powered behavioral analysis to make better hiring decisions.
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <button className="px-8 py-4 bg-white text-black font-medium rounded-lg hover:bg-gray-100 transition-all hover:scale-105 text-lg">
                Start free trial
              </button>
              <button className="px-8 py-4 bg-transparent border-2 border-white text-white font-medium rounded-lg hover:bg-white hover:text-black transition-all text-lg">
                Book a demo
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-6 py-16 max-w-7xl mx-auto border-t border-white/10">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center font-bold text-sm">
                AI
              </div>
              <span className="text-xl font-bold">InterviewAR</span>
            </div>
            <p className="text-sm text-gray-500">
              AI-powered interview intelligence for the modern workforce.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Product</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Security</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Integrations</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Company</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="#" className="hover:text-white transition-colors">About</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Support</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Status</a></li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-gray-500">
          <p>© 2026 InterviewAR. All rights reserved.</p>
          <div className="flex gap-6">
            <a href="#" className="hover:text-white transition-colors">Privacy</a>
            <a href="#" className="hover:text-white transition-colors">Terms</a>
            <a href="#" className="hover:text-white transition-colors">Cookies</a>
          </div>
        </div>
      </footer>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-20px);
          }
        }

        @keyframes pulse-slow {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }

        @keyframes slide-in {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.8s ease-out;
        }

        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out;
          animation-fill-mode: both;
        }

        .animate-float {
          animation: float 6s ease-in-out infinite;
        }

        .animate-pulse-slow {
          animation: pulse-slow 2s ease-in-out infinite;
        }

        .animate-slide-in {
          animation: slide-in 0.5s ease-out;
          animation-fill-mode: both;
        }

        .fade-in-section {
          opacity: 0;
          transform: translateY(30px);
          transition: opacity 0.8s ease-out, transform 0.8s ease-out;
        }

        .fade-in-section.animate-in {
          opacity: 1;
          transform: translateY(0);
        }
      `}</style>
    </div>
  );
}