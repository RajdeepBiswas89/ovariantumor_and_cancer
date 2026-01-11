
import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, ShieldAlert, FileText, CheckCircle, Loader2, BrainCircuit, Microscope, Zap, Database, Search } from 'lucide-react';
import { DiagnosisClass, AnalysisResult, PatientInfo } from '../types';
import ReportTemplate from './ReportTemplate';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const DiagnosticTool: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [step, setStep] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    name: '',
    age: '',
    gender: 'Female',
    history: ''
  });
  const [showReport, setShowReport] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const steps = [
    "Image Pre-processing",
    "ResNet18 Texture Analysis",
    "ResNet50 Semantic Extraction",
    "Hybrid Decision Fusion",
    "Clinical Validation"
  ];

  useEffect(() => {
    let interval: any;
    if (analyzing) {
      interval = setInterval(() => {
        setStep(s => (s < steps.length - 1 ? s + 1 : s));
      }, 1500);
    } else {
      setStep(0);
    }
    return () => clearInterval(interval);
  }, [analyzing]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const runHybridAnalysis = async () => {
    if (!image) return;
    setAnalyzing(true);
    
    try {
      const blob = await (await fetch(image)).blob();
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');

      const response = await fetch('http://localhost:8002/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend API error: ${response.statusText}`);
      }

      const parsedData = await response.json();
      
      setResult({
        diagnosis: parsedData.Diagnosis as DiagnosisClass,
        confidence: parsedData['Confidence Score'] || 0.95,
        findings: parsedData['Clinical Findings'] || 'Inference successful.',
        recommendations: parsedData.Recommendations || [],
        timestamp: new Date().toLocaleString(),
        patientId: `OVA-${Math.floor(1000 + Math.random() * 9000)}`,
        patientName: patientInfo.name || 'Anonymous Patient'
      });
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Clinical inference interrupted.');
    } finally {
      setTimeout(() => setAnalyzing(false), 2000); // Allow animation to breathe
    }
  };

  const chartData = result ? [
    { name: 'Confidence', value: result.confidence * 100 },
    { name: 'Residual', value: (1 - result.confidence) * 100 }
  ] : [];

  return (
    <div className="max-w-7xl mx-auto px-4 py-16">
      <div className="grid lg:grid-cols-12 gap-12">
        {/* Left Control Panel */}
        <div className="lg:col-span-4 space-y-8">
          <div className="bg-white rounded-3xl p-8 border border-slate-200 shadow-xl shadow-slate-200/50">
            <h3 className="text-xs font-black text-slate-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
              <Database size={14} className="text-teal-600" />
              Ingestion Parameters
            </h3>
            <div className="space-y-5">
              <InputField label="Patient ID" placeholder="Auto-generated" disabled value={result?.patientId || ''} />
              <InputField 
                label="Full Name" 
                placeholder="Required" 
                value={patientInfo.name} 
                onChange={v => setPatientInfo({...patientInfo, name: v})} 
              />
              <InputField 
                label="Subject Age" 
                placeholder="YY" 
                type="number"
                value={patientInfo.age} 
                onChange={v => setPatientInfo({...patientInfo, age: v})} 
              />
            </div>
          </div>

          <div className="bg-slate-900 rounded-3xl p-8 text-white">
            <h3 className="text-xs font-black text-teal-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
              <Zap size={14} />
              Inference Pipeline
            </h3>
            <div className="space-y-4">
              {steps.map((s, i) => (
                <div key={i} className={`flex items-center gap-4 transition-all duration-500 ${step >= i ? 'opacity-100' : 'opacity-30'}`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-black border ${step > i ? 'bg-teal-500 border-teal-500 text-white' : 'border-slate-700 text-slate-500'}`}>
                    {step > i ? <CheckCircle size={12} /> : i + 1}
                  </div>
                  <span className="text-xs font-bold uppercase tracking-wider">{s}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Center Analysis Area */}
        <div className="lg:col-span-8 space-y-12">
          <div className="relative aspect-video bg-black rounded-[2.5rem] overflow-hidden shadow-2xl group border-8 border-white">
            {image ? (
              <>
                <img src={image} className="w-full h-full object-contain opacity-80" alt="Clinical Scan" />
                {analyzing && <div className="scan-line"></div>}
                
                {/* Simulated Heatmap Overlay on Hover or Result */}
                {(analyzing || result) && (
                  <div className="absolute inset-0 heatmap-overlay pointer-events-none">
                     <div className="absolute top-1/3 left-1/4 w-32 h-32 rounded-full bg-red-500"></div>
                     <div className="absolute bottom-1/2 right-1/3 w-40 h-40 rounded-full bg-orange-400"></div>
                  </div>
                )}
                
                <div className="absolute top-6 left-6 flex gap-2">
                  <div className="px-3 py-1 bg-black/50 backdrop-blur-md rounded-lg text-[10px] font-black text-teal-400 uppercase tracking-widest flex items-center gap-2">
                    <Microscope size={14} />
                    Active Scan Area
                  </div>
                </div>
                
                <button 
                  onClick={() => { setImage(null); setResult(null); }}
                  className="absolute bottom-6 right-6 p-3 bg-white/10 backdrop-blur-xl text-white rounded-2xl hover:bg-red-500 transition-all opacity-0 group-hover:opacity-100"
                >
                  <X size={20} />
                </button>
              </>
            ) : (
              <div 
                onClick={() => fileInputRef.current?.click()}
                className="w-full h-full flex flex-col items-center justify-center cursor-pointer hover:bg-slate-900 transition-colors"
              >
                <div className="w-20 h-20 rounded-full bg-slate-800 flex items-center justify-center text-teal-500 mb-6 pulse-soft">
                  <Upload size={32} />
                </div>
                <h4 className="text-xl font-bold text-white mb-2 tracking-tight">Ingest Medical Imaging</h4>
                <p className="text-slate-500 text-sm">DICOM • MRI • CT • Ultrasound</p>
              </div>
            )}
            <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleImageUpload} />
          </div>

          <div className="flex gap-4">
            <button 
              disabled={!image || analyzing}
              onClick={runHybridAnalysis}
              className={`flex-grow py-6 rounded-2xl font-black text-xl flex items-center justify-center gap-3 transition-all ${
                !image || analyzing ? 'bg-slate-200 text-slate-400 cursor-not-allowed' : 'bg-teal-600 text-white shadow-xl shadow-teal-600/20 hover:scale-[1.02] hover:bg-teal-700'
              }`}
            >
              {analyzing ? <Loader2 className="animate-spin" /> : <BrainCircuit size={28} />}
              {analyzing ? `Executing ${steps[step]}...` : 'Execute Neural Diagnostic'}
            </button>
            <button className="px-8 bg-white border border-slate-200 rounded-2xl flex items-center justify-center text-slate-700 hover:bg-slate-50 transition-all">
               <Search size={24} />
            </button>
          </div>

          {result && (
            <div className="animate-fade-in bg-white rounded-[2.5rem] p-10 border border-teal-100 shadow-2xl grid md:grid-cols-2 gap-12">
              <div className="space-y-8">
                <div>
                   <span className="inline-block px-3 py-1 bg-teal-100 text-teal-700 text-[10px] font-black uppercase tracking-widest rounded-full mb-3">
                     Diagnostic Confirmed
                   </span>
                   <h3 className="text-4xl font-black text-slate-900 leading-tight">
                    {result.diagnosis}
                   </h3>
                </div>
                <div className="space-y-4">
                  <h4 className="text-xs font-black text-slate-400 uppercase tracking-widest">Clinical Commentary</h4>
                  <p className="text-slate-600 leading-relaxed font-medium bg-slate-50 p-6 rounded-2xl border border-slate-100 italic">
                    "{result.findings}"
                  </p>
                </div>
              </div>

              <div className="flex flex-col justify-between">
                <div className="h-48 w-full">
                   <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                         <Pie data={chartData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={10} dataKey="value">
                            <Cell fill="#0d9488" stroke="none" />
                            <Cell fill="#f1f5f9" stroke="none" />
                         </Pie>
                         <Tooltip />
                      </PieChart>
                   </ResponsiveContainer>
                   <div className="text-center -mt-28 mb-12">
                      <p className="text-3xl font-black text-slate-900">{(result.confidence * 100).toFixed(1)}%</p>
                      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Inference Prob.</p>
                   </div>
                </div>
                <button 
                  onClick={() => setShowReport(true)}
                  className="w-full bg-slate-900 text-white py-5 rounded-2xl font-bold flex items-center justify-center gap-3 hover:bg-slate-800 transition-all"
                >
                  <FileText size={20} />
                  Assemble Clinical Report
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {showReport && result && (
        <ReportTemplate 
          result={result} 
          patientInfo={patientInfo} 
          image={image || ''} 
          onClose={() => setShowReport(false)} 
        />
      )}
    </div>
  );
};

const InputField: React.FC<{ label: string; placeholder: string; value: string; onChange?: (v: string) => void; disabled?: boolean; type?: string }> = ({ label, placeholder, value, onChange, disabled, type = 'text' }) => (
  <div className="space-y-1.5">
    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest pl-1">{label}</label>
    <input 
      type={type}
      disabled={disabled}
      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 focus:ring-2 focus:ring-teal-500 outline-none text-sm font-bold placeholder:text-slate-300 disabled:opacity-50"
      placeholder={placeholder}
      value={value}
      onChange={e => onChange?.(e.target.value)}
    />
  </div>
);

export default DiagnosticTool;
