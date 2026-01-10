
import React from 'react';
import { Network, Layers, Database, Cpu } from 'lucide-react';

const ResearchSection: React.FC = () => {
  return (
    <section className="py-24 bg-white border-y border-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 mb-4">The Science Behind OvaScan</h2>
          <p className="text-slate-500 max-w-2xl mx-auto">
            Our proprietary hybrid architecture combines shallow and deep residual networks to capture both intricate texture patterns and broad morphological structures.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          <ResearchCard 
            icon={<Layers className="text-teal-600" />}
            title="ResNet50 Backbone"
            desc="Deeper feature extraction layers focused on capturing semantic medical patterns and stage-specific indicators."
          />
          <ResearchCard 
            icon={<Network className="text-blue-600" />}
            title="ResNet18 Fusion"
            desc="Efficient, shallow residual blocks ensure high-frequency details (like edges and inflammatory markers) aren't lost."
          />
          <ResearchCard 
            icon={<Database className="text-purple-600" />}
            title="Curated Dataset"
            desc="Trained on over 150,000 anonymized ovarian ultrasound and MRI samples with histopathological verification."
          />
          <ResearchCard 
            icon={<Cpu className="text-slate-700" />}
            title="Zero-Latency Inference"
            desc="Optimized weights allow for near-instant classification on standard mobile and desktop environments."
          />
        </div>

        <div className="mt-20 glass-morphism p-12 rounded-[2rem] border border-teal-100">
          <div className="lg:flex gap-12 items-center">
            <div className="lg:w-1/3 mb-10 lg:mb-0">
               <div className="relative">
                 <div className="bg-teal-600 h-64 w-full rounded-3xl flex items-center justify-center text-white overflow-hidden">
                    <div className="flex flex-col items-center gap-4">
                        <div className="flex gap-2">
                            {[1,2,3,4].map(i => <div key={i} className="w-2 h-12 bg-white/20 rounded-full animate-pulse" style={{animationDelay: `${i*100}ms`}}></div>)}
                        </div>
                        <p className="font-mono text-xs opacity-60">HYBRID_LAYER_ACTIVATION</p>
                    </div>
                 </div>
                 <div className="absolute -bottom-6 -right-6 bg-white p-4 rounded-2xl shadow-xl border border-slate-100">
                    <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-1">Hybrid Accuracy</p>
                    <p className="text-3xl font-black text-teal-600">98.2<span className="text-sm">%</span></p>
                 </div>
               </div>
            </div>
            <div className="lg:w-2/3">
              <h3 className="text-2xl font-bold text-slate-900 mb-6">Multi-Class Classification Protocol</h3>
              <div className="grid sm:grid-cols-2 gap-6">
                <ProtocolItem title="Infected Class" desc="Detects pelvic inflammatory disease (PID) and acute tubovarian abscesses." />
                <ProtocolItem title="Benign Tumor" desc="Identification of serous/mucinous cystadenomas and dermoid cysts." />
                <ProtocolItem title="Malignant" desc="Specific markers for high-grade serous ovarian carcinoma (HGSOC)." />
                <ProtocolItem title="Healthy" desc="Baseline normal ovarian volume and stromal appearance validation." />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const ResearchCard: React.FC<{ icon: React.ReactNode; title: string; desc: string }> = ({ icon, title, desc }) => (
  <div className="p-8 rounded-3xl bg-slate-50 hover:bg-white hover:shadow-xl hover:shadow-slate-200/50 transition-all border border-transparent hover:border-slate-100 group">
    <div className="mb-6 bg-white p-4 rounded-2xl w-fit shadow-sm group-hover:scale-110 transition-transform">
      {icon}
    </div>
    <h4 className="text-xl font-bold text-slate-900 mb-3">{title}</h4>
    <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
  </div>
);

const ProtocolItem: React.FC<{ title: string; desc: string }> = ({ title, desc }) => (
  <div className="flex gap-4">
    <div className="shrink-0 w-2 h-2 rounded-full bg-teal-500 mt-2"></div>
    <div>
      <h5 className="font-bold text-slate-800">{title}</h5>
      <p className="text-sm text-slate-500">{desc}</p>
    </div>
  </div>
);

export default ResearchSection;
