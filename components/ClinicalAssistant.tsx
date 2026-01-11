
import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, MessageSquare, Play, Info, Activity } from 'lucide-react';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Add type definition for Vite's import.meta.env
declare global {
  interface ImportMeta {
    env: {
      VITE_GEMINI_API_KEY?: string;
    };
  }
}

const ClinicalAssistant: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  // Initialize Google Gemini AI
  const getGeminiAPI = () => {
    try {
      const apiKey = import.meta.env.VITE_GEMINI_API_KEY || '';
      console.log('API Key from env:', apiKey ? `Key exists (${apiKey.substring(0, 5)}...)` : 'Key is empty or undefined');
      
      if (!apiKey) {
        console.warn('Gemini API key not found. Voice assistant will work but AI responses may be limited.');
        return null;
      }
      
      // Initialize with explicit API version
      const genAI = new GoogleGenerativeAI(apiKey);
      console.log('GoogleGenerativeAI initialized successfully');
      return genAI;
    } catch (error) {
      console.error('Error initializing GoogleGenerativeAI:', error);
      return null;
    }
  };

  // Initialize Speech Recognition - setup once on mount
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = async (event: SpeechRecognitionEvent) => {
        const last = event.results.length - 1;
        const text = event.results[last][0].transcript;
        
        setTranscript(prev => [...prev, `You: ${text}`]);
        setIsProcessing(true);

        try {
          // Get AI response from Gemini
          const gemini = getGeminiAPI();
          let aiResponse = "I'm sorry, I couldn't process that request. Please check your API key configuration.";

          if (gemini) {
            try {
              console.log('Initializing model...');
              const model = gemini.getGenerativeModel({ 
                model: 'gemini-1.5-flash',
                generationConfig: {
                  temperature: 0.9,
                  topK: 1,
                  topP: 1,
                  maxOutputTokens: 2048,
                },
              });
              
              const prompt = `You are a clinical assistant specialized in ovarian pathology. Answer this question professionally and concisely: ${text}`;
              console.log('Sending request to Gemini API...');
              
              const result = await model.generateContent(prompt);
              const response = await result.response;
              aiResponse = response.text();
              console.log('Received response from Gemini API');
            } catch (error) {
              console.error('Error getting AI response:', error);
              aiResponse = "I'm having trouble connecting to the AI service. Please try again later.";
            }
          } else {
            // Fallback response if no API key
            aiResponse = handleFallbackResponse(text);
          }

          setTranscript(prev => [...prev, `AI: ${aiResponse}`]);
          
          // Speak the response
          speakText(aiResponse);
        } catch (error) {
          console.error('Error getting AI response:', error);
          const errorMsg = "I encountered an error processing your request. Please try again.";
          setTranscript(prev => [...prev, `AI: ${errorMsg}`]);
          speakText(errorMsg);
        } finally {
          setIsProcessing(false);
        }
      };

      recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('Speech recognition error:', event.error);
        setIsProcessing(false);
        if (event.error === 'no-speech') {
          setTranscript(prev => [...prev, 'AI: No speech detected. Please try again.']);
        } else if (event.error === 'network') {
          setTranscript(prev => [...prev, 'AI: Network error. Please check your connection.']);
        }
      };

      recognitionRef.current.onend = () => {
        // Recognition ended - will be restarted in toggleAssistant if needed
      };
    } else {
      console.warn('Speech Recognition API not supported in this browser');
    }

    synthRef.current = window.speechSynthesis;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []); // Empty dependency array - initialize once

  const handleFallbackResponse = (text: string): string => {
    const lowerText = text.toLowerCase();
    if (lowerText.includes('guideline') || lowerText.includes('who') || lowerText.includes('staging')) {
      return "According to WHO guidelines for ovarian pathology, classification follows standardized criteria based on histological features, molecular markers, and clinical presentation. For FIGO staging, ovarian cancers are staged from I to IV based on extent of disease.";
    } else if (lowerText.includes('classification') || lowerText.includes('rationale')) {
      return "Our hybrid ResNet50-ResNet18 model classifies ovarian tissue using deep learning features. The model extracts texture and semantic features, applies recursive feature elimination, and fuses predictions using Kalman filtering and Dempster-Shafer theory for robust classification.";
    } else if (lowerText.includes('hello') || lowerText.includes('hi')) {
      return "Hello! I'm your clinical assistant. How can I help you with ovarian pathology questions today?";
    } else {
      return "I understand your question. For detailed clinical guidance, please configure your Gemini API key for enhanced AI responses. I can help with WHO guidelines, FIGO staging, and classification rationales.";
    }
  };

  const speakText = (text: string) => {
    if (synthRef.current) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;
      synthRef.current.speak(utterance);
    }
  };

  const toggleAssistant = () => {
    if (!isActive) {
      // Starting
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start();
          setIsActive(true);
          setTranscript(prev => [...prev, "AI: ResNet Clinical Assistant Online. How can I assist with your findings today?"]);
          speakText("ResNet Clinical Assistant Online. How can I assist with your findings today?");
        } catch (error) {
          console.error('Error starting recognition:', error);
          setTranscript(prev => [...prev, "Error: Could not start voice recognition. Please check your browser permissions."]);
        }
      } else {
        setTranscript(prev => [...prev, "Error: Speech Recognition is not supported in your browser."]);
      }
    } else {
      // Stopping
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
      setIsActive(false);
      setTranscript(prev => [...prev, "AI: Assistant offline."]);
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
