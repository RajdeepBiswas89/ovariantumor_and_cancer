
import React from 'react';
import { Stethoscope, Pill, Apple, BookOpen, Search, PhoneCall, ShieldCheck, Target } from 'lucide-react';

const Solutions: React.FC = () => {
  return (
    <section className="py-32 bg-slate-950 text-white relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none surgical-grid"></div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-20 gap-8">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-teal-500/10 text-teal-400 rounded-full text-xs font-bold uppercase tracking-[0.2em] mb-6">
              <ShieldCheck size={14} />
              Standardized Clinical Pathways
            </div>
            <h2 className="text-5xl font-black mb-6 leading-tight">Precision <span className="text-teal-400 italic">Therapeutic</span> Integration</h2>
            <p className="text-slate-400 text-xl leading-relaxed">
              Moving beyond classification into evidence-based medical management aligned with global oncology benchmarks.
            </p>
          </div>
          <button className="bg-white text-slate-900 px-10 py-5 rounded-2xl font-black uppercase tracking-widest text-sm hover:bg-teal-400 transition-all shadow-2xl">
            Institutional Portal
          </button>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <SolutionCard 
            icon={<Target className="text-teal-400" />}
            title="Surgical Management"
            subtitle="Stage I - IV Protocol"
            points={[
              "Comprehensive Cytoreductive Surgery",
              "Lymphadenectomy & Mapping",
              "Minimally Invasive Laparoscopy",
              "Hyperthermic Chemotherapy (HIPEC)"
            ]}
          />
          <SolutionCard 
            icon={<Pill className="text-blue-400" />}
            title="Precision Therapeutics"
            subtitle="Genomic-Led Interventions"
            points={[
              "PARP Inhibitors for BRCA+ Cases",
              "Anti-Angiogenic Biologicals",
              "Targeted Immuno-Oncology",
              "Platinum-Based Adjuvant Cycles"
            ]}
          />
          <SolutionCard 
            icon={<Stethoscope className="text-purple-400" />}
            title="Follow-up Oncology"
            subtitle="Surveillance Standards"
            points={[
              "Biannual CA-125 Serum Analysis",
              "ResNet-Assisted Serial Imaging",
              "Genetic Risk Counseling",
              "Survivorship Quality Metrics"
            ]}
          />
        </div>

        <div className="mt-24 p-[2px] bg-gradient-to-r from-teal-500 via-blue-500 to-purple-600 rounded-[3rem]">
            <div className="bg-slate-950 rounded-[2.9rem] p-16 text-center relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-teal-500/10 rounded-full blur-[80px]"></div>
                <h3 className="text-3xl font-black mb-8">Establish Institutional Connectivity</h3>
                <div className="flex flex-wrap justify-center gap-12">
                    <button className="group flex items-center gap-4 text-xl font-bold hover:text-teal-400 transition-all">
                        <div className="w-12 h-12 rounded-xl bg-slate-900 flex items-center justify-center group-hover:bg-teal-600 transition-colors">
                            <PhoneCall size={24} />
                        </div>
                        Oncology Crisis Line
                    </button>
                    <button className="group flex items-center gap-4 text-xl font-bold hover:text-teal-400 transition-all">
                        <div className="w-12 h-12 rounded-xl bg-slate-900 flex items-center justify-center group-hover:bg-teal-600 transition-colors">
                            <Search size={24} />
                        </div>
                        Diagnostic Network
                    </button>
                </div>
            </div>
        </div>
      </div>
    </section>
  );
};

const SolutionCard: React.FC<{ icon: React.ReactNode; title: string; subtitle: string; points: string[] }> = ({ icon, title, subtitle, points }) => (
  <div className="p-10 rounded-[2.5rem] bg-slate-900/50 border border-slate-800 hover:border-teal-500/50 hover:bg-slate-900 transition-all group">
    <div className="mb-8 p-5 rounded-2xl bg-slate-800 w-fit group-hover:scale-110 transition-transform">
      {icon}
    </div>
    <div className="mb-8">
      <h4 className="text-2xl font-black mb-1">{title}</h4>
      <p className="text-[10px] font-black text-teal-400 uppercase tracking-[0.3em]">{subtitle}</p>
    </div>
    <ul className="space-y-5">
      {points.map((p, i) => (
        <li key={i} className="flex items-start gap-4 text-slate-400 group-hover:text-slate-300">
          <div className="w-2 h-2 rounded-full bg-teal-600 mt-2 shrink-0 shadow-[0_0_10px_rgba(13,148,136,0.5)]"></div>
          <span className="text-sm font-medium leading-relaxed">{p}</span>
        </li>
      ))}
    </ul>
  </div>
);

export default Solutions;
