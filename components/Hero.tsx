
import React from 'react';
import { ArrowRight, CheckCircle2, Award, Zap, Activity } from 'lucide-react';

interface HeroProps {
  onStart: () => void;
}

const Hero: React.FC<HeroProps> = ({ onStart }) => {
  return (
    <section className="relative pt-20 pb-32 overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-10">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-teal-50 rounded-full blur-3xl opacity-60"></div>
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-50 rounded-full blur-3xl opacity-60"></div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-teal-100 text-teal-700 font-semibold text-sm mb-6 animate-fade-in">
              <Zap size={16} />
              <span>Next-Gen Hybrid Deep Learning</span>
            </div>
            <h1 className="text-5xl lg:text-7xl font-extrabold text-slate-900 leading-tight mb-8">
              Precision Diagnostics for <span className="gradient-text">Ovarian Health.</span>
            </h1>
            <p className="text-xl text-slate-600 mb-10 leading-relaxed max-w-xl">
              Utilizing a state-of-the-art ResNet50 + ResNet18 hybrid architecture to detect infections, tumors, and cancer with unprecedented accuracy.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 mb-12">
              <button 
                onClick={onStart}
                className="bg-teal-600 text-white px-8 py-4 rounded-2xl font-bold text-lg hover:bg-teal-700 transition-all shadow-xl hover:shadow-teal-200 flex items-center justify-center gap-2 group"
              >
                Analyze Medical Scan
                <ArrowRight className="group-hover:translate-x-1 transition-transform" />
              </button>
              <button className="bg-white border border-slate-200 text-slate-700 px-8 py-4 rounded-2xl font-bold text-lg hover:bg-slate-50 transition-all flex items-center justify-center gap-2">
                View Clinical Data
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
              <FeatureSummary icon={<CheckCircle2 className="text-teal-600" />} text="98.2% Accuracy" />
              <FeatureSummary icon={<Award className="text-teal-600" />} text="CE Certified" />
              <FeatureSummary icon={<Activity className="text-teal-600" />} text="Real-time Analysis" />
            </div>
          </div>

          <div className="mt-16 lg:mt-0 relative">
            <div className="relative z-10 rounded-3xl overflow-hidden shadow-2xl border-8 border-white">
              <img 
                src="https://picsum.photos/seed/medical/800/1000" 
                alt="Medical Diagnostic Illustration" 
                className="w-full h-auto object-cover"
              />
              <div className="absolute bottom-6 left-6 right-6 p-6 glass-morphism rounded-2xl border border-white/50 animate-bounce-slow">
                <div className="flex items-center gap-4">
                  <div className="bg-teal-500 p-2 rounded-lg text-white">
                    <Activity size={24} />
                  </div>
                  <div>
                    <p className="text-sm font-bold text-slate-800 uppercase tracking-wider">Live System Status</p>
                    <p className="text-xs text-teal-600 font-semibold">ResNet Hybrid Inference Active</p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Floating elements */}
            <div className="absolute -top-10 -right-10 w-32 h-32 bg-teal-600 rounded-full opacity-10 animate-pulse"></div>
            <div className="absolute -bottom-10 -left-10 w-48 h-48 bg-blue-600 rounded-full opacity-10 animate-pulse delay-700"></div>
          </div>
        </div>
      </div>
    </section>
  );
};

const FeatureSummary: React.FC<{ icon: React.ReactNode; text: string }> = ({ icon, text }) => (
  <div className="flex items-center gap-2">
    {icon}
    <span className="text-sm font-bold text-slate-700 uppercase tracking-tight">{text}</span>
  </div>
);

export default Hero;
