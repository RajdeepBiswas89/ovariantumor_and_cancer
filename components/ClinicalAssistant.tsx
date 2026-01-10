
import React, { useState, useRef } from 'react';
import { GoogleGenAI, Modality } from '@google/genai';
import { Mic, MicOff, MessageSquare, Play, Info, Activity } from 'lucide-react';

const ClinicalAssistant: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Simplified simulation for display purposes, but hooks into industry standards
  const toggleAssistant = () => {
    setIsActive(!isActive);
    if (!isActive) {
      setTranscript(prev => [...prev, "AI: ResNet Clinical Assistant Online. How can I assist with your findings today?"]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <div className="bg-slate-900 rounded-[3rem] p-12 text-white shadow-2xl relative overflow-hidden">
        {/* Background Aura */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-teal-500/10 rounded-full blur-[100px]"></div>
        
        <div className="relative z-10 flex flex-col items-center text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-teal-500/20 text-teal-400 font-bold text-xs mb-8 uppercase tracking-[0.2em]">
            <Activity size={14} />
            Handoff Protocol Active
          </div>
          
          <h2 className="text-4xl font-black mb-4">Hands-Free Clinical Review</h2>
          <p className="text-slate-400 max-w-lg mb-12">
            Speak to the AI to query WHO guidelines, FIGO staging, or request classification rationales while performing patient examinations.
          </p>

          <div className="relative mb-12">
            <button 
              onClick={toggleAssistant}
              className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-500 border-8 ${
                isActive ? 'bg-teal-600 border-teal-500 shadow-[0_0_60px_rgba(13,148,136,0.4)] scale-110' : 'bg-slate-800 border-slate-700 hover:bg-slate-700'
              }`}
            >
              {isActive ? <Mic size={48} className="animate-pulse" /> : <MicOff size={48} className="text-slate-500" />}
            </button>
            {isActive && (
              <div className="absolute inset-0 rounded-full border border-teal-500/50 animate-ping"></div>
            )}
          </div>

          <div className="w-full bg-black/40 backdrop-blur-md rounded-3xl p-8 border border-white/5 text-left h-64 overflow-y-auto font-mono text-sm space-y-4 surgical-grid">
            {transcript.length === 0 ? (
              <p className="text-slate-600 italic">Waiting for voice input command...</p>
            ) : (
              transcript.map((line, i) => (
                <div key={i} className={`${line.startsWith('AI:') ? 'text-teal-400' : 'text-slate-300'}`}>
                  {line}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mt-12">
        <AssistanceFeature icon={<MessageSquare />} label="Guideline Query" desc="Ask about latest ovarian pathology standards." />
        <AssistanceFeature icon={<Play />} label="Step-by-Step" desc="Guided collection protocol for biopsy samples." />
        <AssistanceFeature icon={<Info />} label="Logic Trace" desc="Voice-triggered reasoning for Hybrid AI decisions." />
      </div>
    </div>
  );
};

const AssistanceFeature: React.FC<{ icon: React.ReactNode; label: string; desc: string }> = ({ icon, label, desc }) => (
  <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-all">
    <div className="p-3 bg-slate-50 text-teal-600 rounded-xl w-fit mb-4">{icon}</div>
    <h4 className="font-bold text-slate-900 mb-1">{label}</h4>
    <p className="text-xs text-slate-500 font-medium">{desc}</p>
  </div>
);

export default ClinicalAssistant;
