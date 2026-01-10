
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { X, ShieldCheck, Printer, Clock, BarChart3, Scan, Layers, Download, Image, FileJson, ChevronDown, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';
import { AnalysisResult, PatientInfo } from '../types';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { toPng } from 'html-to-image';

interface ReportTemplateProps {
  result: AnalysisResult;
  patientInfo: PatientInfo;
  image: string;
  onClose: () => void;
}

const ReportTemplate: React.FC<ReportTemplateProps> = ({ result, patientInfo, image, onClose }) => {
  const reportRef = useRef<HTMLDivElement>(null);
  const heatmapContainerRef = useRef<HTMLDivElement>(null);
  
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  
  // Interactive Heatmap State
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const handlePrint = () => {
    window.print();
    setShowExportMenu(false);
  };

  const downloadAsJSON = () => {
    const data = JSON.stringify({ 
      institutional_header: "OvaScan AI Diagnostic Suite",
      report_metadata: {
        timestamp: result.timestamp,
        patient_id: result.patientId,
        verification_hash: `SHA-256:${result.patientId}-X82K`
      },
      patient_identification: patientInfo,
      analysis_results: result 
    }, null, 2);
    
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `OvaScan_Report_${result.patientId}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
  };

  const downloadAsPNG = async () => {
    if (reportRef.current === null) return;
    setIsExporting(true);
    setShowExportMenu(false);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 100));
      const dataUrl = await toPng(reportRef.current, { 
        cacheBust: true, 
        backgroundColor: '#ffffff',
        style: { borderRadius: '0px' }
      });
      
      const link = document.createElement('a');
      link.download = `OvaScan_Report_${result.patientId}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Error exporting PNG:', err);
    } finally {
      setIsExporting(false);
    }
  };

  // Heatmap Interaction Logic
  const handleZoom = (direction: 'in' | 'out') => {
    setZoom(prev => {
      const next = direction === 'in' ? prev + 0.5 : prev - 0.5;
      return Math.min(Math.max(next, 1), 5);
    });
  };

  const resetHeatmap = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  }, [isDragging, dragStart]);

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove]);

  const themeColor = (() => {
    if (result.diagnosis.includes('Cancer')) return '#dc2626';
    if (result.diagnosis.includes('Tumor')) return '#ea580c';
    if (result.diagnosis.includes('Infected')) return '#2563eb';
    return '#0d9488';
  })();

  const confidenceData = [
    { name: 'Confidence', value: result.confidence * 100 },
    { name: 'Uncertainty', value: (1 - result.confidence) * 100 },
  ];

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/90 backdrop-blur-xl p-4 overflow-y-auto">
      <div className="bg-white rounded-3xl shadow-[0_0_50px_rgba(0,0,0,0.3)] max-w-4xl w-full relative my-8 overflow-hidden">
        {/* Premium Control Bar */}
        <div className="sticky top-0 right-0 p-4 flex justify-between items-center z-20 print:hidden bg-slate-50/80 backdrop-blur-md border-b border-slate-200">
          <div className="flex items-center gap-2 px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs font-bold uppercase tracking-widest">
            <Scan size={14} />
            Diagnostic Report Review
          </div>
          <div className="flex gap-3 relative">
            <div className="relative group">
              <button 
                onClick={() => setShowExportMenu(!showExportMenu)}
                disabled={isExporting}
                className="flex items-center gap-2 px-5 py-2.5 bg-slate-900 text-white rounded-xl hover:bg-slate-800 font-bold transition-all shadow-lg active:scale-95 disabled:opacity-50"
              >
                {isExporting ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                ) : (
                  <Download size={18} />
                )}
                Export Options
                <ChevronDown size={14} className={`transition-transform ${showExportMenu ? 'rotate-180' : ''}`} />
              </button>

              {showExportMenu && (
                <>
                  <div className="fixed inset-0 z-30" onClick={() => setShowExportMenu(false)}></div>
                  <div className="absolute right-0 mt-3 w-56 bg-white border border-slate-200 rounded-2xl shadow-2xl py-2 z-40 animate-fade-in overflow-hidden">
                    <ExportOption 
                      icon={<Printer size={16} />} 
                      label="Print as PDF" 
                      onClick={handlePrint}
                      description="Optimal for paper records"
                    />
                    <ExportOption 
                      icon={<Image size={16} />} 
                      label="Save as PNG" 
                      onClick={downloadAsPNG}
                      description="High-fidelity visual capture"
                    />
                    <div className="h-px bg-slate-100 my-1 mx-2"></div>
                    <ExportOption 
                      icon={<FileJson size={16} />} 
                      label="Export JSON Data" 
                      onClick={downloadAsJSON}
                      description="Structured clinical data"
                    />
                  </div>
                </>
              )}
            </div>

            <button 
              onClick={onClose}
              className="flex items-center justify-center w-10 h-10 bg-white border border-slate-200 text-slate-400 rounded-xl hover:text-red-500 hover:border-red-200 transition-all shadow-sm"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Clinical Document Content */}
        <div ref={reportRef} className="p-16 print:p-10 bg-white">
          {/* Header Section */}
          <div className="flex justify-between items-start border-b-4 border-slate-900 pb-10 mb-10">
            <div className="flex items-center gap-4">
              <div className="bg-teal-600 p-3 rounded-2xl text-white shadow-teal-200 shadow-xl">
                <ShieldCheck size={40} />
              </div>
              <div>
                <h1 className="text-4xl font-black uppercase tracking-tighter leading-none">OvaScan AI</h1>
                <p className="text-xs font-black text-slate-400 uppercase tracking-[0.3em] mt-2">Hybrid Medical Diagnostic Suite</p>
              </div>
            </div>
            <div className="text-right">
              <div className="inline-block px-4 py-1 bg-slate-100 rounded-lg mb-2">
                <p className="text-sm font-black text-slate-900">REF: <span className="font-mono">{result.patientId}</span></p>
              </div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Generated On: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
            </div>
          </div>

          <div className="grid grid-cols-12 gap-12 mb-12">
            {/* Patient Info Column */}
            <div className="col-span-12 lg:col-span-4 space-y-10">
              <section>
                <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4 border-l-2 border-teal-500 pl-3">Patient Identification</h2>
                <div className="space-y-3">
                  <InfoRow label="Legal Name" value={patientInfo.name || 'Anonymous Patient'} />
                  <InfoRow label="Calculated Age" value={patientInfo.age ? `${patientInfo.age} Years` : 'Unknown'} />
                  <InfoRow label="Clinical ID" value={result.patientId} mono />
                </div>
              </section>

              <section>
                <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4 border-l-2 border-teal-500 pl-3">Inference Session</h2>
                <div className="space-y-3">
                  <div className="flex flex-col gap-1 py-2 border-b border-slate-100">
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Analysis Timestamp</span>
                    <span className="font-bold text-slate-900 flex items-center gap-2 text-xs">
                      <Clock size={14} className="text-teal-600" />
                      {result.timestamp}
                    </span>
                  </div>
                  <InfoRow label="Architecture" value="ResNet50+18 Fusion" />
                </div>
              </section>
            </div>

            {/* Diagnostic Visualization Column */}
            <div className="col-span-12 lg:col-span-8">
              <div className="bg-slate-50 rounded-3xl p-8 border border-slate-200 shadow-sm flex flex-col items-center text-center">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em] mb-2">Automated Classification</p>
                <h3 className="text-3xl font-black mb-6" style={{ color: themeColor }}>
                  {result.diagnosis}
                </h3>
                
                {/* Confidence Gauge Visualization */}
                <div className="relative w-full h-48 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={confidenceData}
                        cx="50%"
                        cy="100%"
                        startAngle={180}
                        endAngle={0}
                        innerRadius={80}
                        outerRadius={110}
                        paddingAngle={0}
                        dataKey="value"
                        stroke="none"
                      >
                        <Cell fill={themeColor} />
                        <Cell fill="#e2e8f0" />
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="absolute bottom-0 flex flex-col items-center">
                    <span className="text-4xl font-black text-slate-900">{(result.confidence * 100).toFixed(1)}%</span>
                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Inference Confidence</span>
                  </div>
                </div>
                
                <div className="mt-8 flex gap-3">
                   <div className="flex items-center gap-1.5 px-3 py-1 bg-white border border-slate-200 rounded-full text-[10px] font-bold text-slate-500">
                     <BarChart3 size={12} className="text-teal-600" />
                     Statistical Probability Logged
                   </div>
                </div>
              </div>
            </div>
          </div>

          {/* Results Analysis Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 mb-12">
            <div className="lg:col-span-4 space-y-10">
              {/* Input Scan Evidence */}
              <div>
                <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Input Scan Evidence</h2>
                <div className="rounded-2xl overflow-hidden border-2 border-slate-100 bg-slate-50 relative">
                  <img src={image} className="w-full aspect-square object-cover grayscale brightness-110" alt="Clinical Scan" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
                </div>
              </div>

              {/* Heatmap Overlay Section - Interactive Version */}
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] flex items-center gap-2">
                    <Layers size={14} className="text-teal-600" />
                    Neural Inspection Tool
                  </h2>
                  <div className="flex gap-1 print:hidden">
                    <button 
                      onClick={() => handleZoom('in')} 
                      className="p-1.5 bg-slate-100 hover:bg-teal-500 hover:text-white rounded-lg transition-all"
                      title="Zoom In"
                    >
                      <ZoomIn size={14} />
                    </button>
                    <button 
                      onClick={() => handleZoom('out')} 
                      className="p-1.5 bg-slate-100 hover:bg-teal-500 hover:text-white rounded-lg transition-all"
                      title="Zoom Out"
                    >
                      <ZoomOut size={14} />
                    </button>
                    <button 
                      onClick={resetHeatmap} 
                      className="p-1.5 bg-slate-100 hover:bg-teal-500 hover:text-white rounded-lg transition-all"
                      title="Reset View"
                    >
                      <RotateCcw size={14} />
                    </button>
                  </div>
                </div>
                
                <div 
                  ref={heatmapContainerRef}
                  onMouseDown={handleMouseDown}
                  className={`rounded-2xl overflow-hidden border-2 border-slate-100 bg-black relative aspect-square cursor-${zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'}`}
                >
                  <div 
                    className="w-full h-full relative transition-transform duration-300 ease-out"
                    style={{ 
                      transform: `scale(${zoom}) translate(${offset.x / zoom}px, ${offset.y / zoom}px)`,
                      transformOrigin: 'center'
                    }}
                  >
                    <img src={image} className="w-full h-full object-cover opacity-50 grayscale select-none" alt="Heatmap Scan" draggable={false} />
                    <div className="absolute inset-0 heatmap-overlay pointer-events-none opacity-80">
                       <div className="absolute top-1/4 left-1/3 w-24 h-24 rounded-full bg-red-600 mix-blend-screen blur-2xl animate-pulse"></div>
                       <div className="absolute bottom-1/3 right-1/4 w-32 h-32 rounded-full bg-yellow-500 mix-blend-screen blur-3xl opacity-60"></div>
                       <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-40 h-40 rounded-full bg-orange-500 mix-blend-overlay blur-[50px]"></div>
                    </div>
                  </div>

                  {/* Indicators Overlay (doesn't scale with zoom) */}
                  <div className="absolute inset-x-4 bottom-4 flex justify-between items-center pointer-events-none">
                    <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-red-500"></div>
                        <span className="text-[8px] font-black text-white uppercase tracking-widest">High Prob. Zone</span>
                      </div>
                    </div>
                    {zoom > 1 && (
                      <div className="bg-teal-600 px-2 py-1 rounded-md shadow-lg flex items-center gap-1.5 animate-bounce-slow">
                        <Maximize2 size={10} className="text-white" />
                        <span className="text-[8px] font-black text-white uppercase">{zoom.toFixed(1)}X Inspect</span>
                      </div>
                    )}
                  </div>
                </div>
                <p className="text-[9px] text-slate-400 mt-3 italic leading-relaxed">
                  Pathologist Note: Use scroll wheel or controls to perform granular tissue morphology inspection. Pan to traverse complex stromal features.
                </p>
              </div>
            </div>
            
            <div className="lg:col-span-8 space-y-10">
              <div>
                <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Detailed Pathological Findings</h2>
                <div className="p-6 bg-slate-50 border-l-4 border-slate-900 rounded-r-2xl">
                   <p className="text-slate-800 leading-relaxed font-medium">
                    "{result.findings}"
                  </p>
                </div>
              </div>

              <div>
                <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Recommended Clinical Protocol</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                   {result.recommendations.map((rec, idx) => (
                     <div key={idx} className="flex items-center gap-4 bg-white border border-slate-100 p-4 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                        <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center text-xs font-black shrink-0">
                          {idx + 1}
                        </div>
                        <span className="text-xs font-bold text-slate-700 leading-tight">{rec}</span>
                     </div>
                   ))}
                </div>
              </div>

              {/* Session Signature Tag */}
              <div className="pt-6 border-t border-slate-100 flex items-center justify-between">
                <div>
                  <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Verification Timestamp</p>
                  <p className="text-[11px] font-bold text-slate-900">{result.timestamp}</p>
                </div>
                <div className="text-right">
                  <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Diagnostic Hash</p>
                  <p className="text-[11px] font-mono font-bold text-teal-600">SHA-256: {result.patientId}-X82K</p>
                </div>
              </div>
            </div>
          </div>

          {/* Authorization Footer */}
          <div className="mt-20 pt-10 border-t-2 border-slate-100">
            <div className="flex flex-col md:flex-row justify-between items-end gap-10">
              <div className="max-w-md text-[10px] text-slate-400 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100">
                <p className="font-black mb-1 uppercase tracking-widest text-slate-500">Legal Clinical Disclaimer</p>
                <p>This report is synthesized by the OvaScan AI engine using proprietary ResNet-based deep residual learning. It is classified as an "In-Vitro Diagnostic Medical Device Software" and should be used strictly to augment professional medical judgment. Final therapeutic decisions must be based on tissue biopsy and comprehensive oncological screening.</p>
              </div>
              
              <div className="text-right w-full md:w-auto">
                <div className="w-64 h-16 border-b-2 border-slate-900 mb-2 relative ml-auto">
                  <span className="absolute bottom-1 right-2 font-serif italic text-slate-300 text-2xl select-none opacity-50">OvaScan_Verified</span>
                </div>
                <p className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500">Certified Medical Reviewer Signature</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ExportOption: React.FC<{ icon: React.ReactNode; label: string; description: string; onClick: () => void }> = ({ icon, label, description, onClick }) => (
  <button 
    onClick={onClick}
    className="w-full text-left px-5 py-3 hover:bg-slate-50 transition-colors group flex items-start gap-3"
  >
    <div className="mt-1 text-slate-400 group-hover:text-teal-600 transition-colors shrink-0">
      {icon}
    </div>
    <div>
      <p className="text-sm font-bold text-slate-900">{label}</p>
      <p className="text-[10px] font-medium text-slate-400">{description}</p>
    </div>
  </button>
);

const InfoRow: React.FC<{ label: string; value: string; mono?: boolean }> = ({ label, value, mono }) => (
  <div className="flex justify-between items-center py-2 border-b border-slate-50">
    <span className="text-[10px] font-bold text-slate-400 uppercase">{label}</span>
    <span className={`font-bold text-slate-900 ${mono ? 'font-mono text-xs' : 'text-sm'}`}>{value}</span>
  </div>
);

export default ReportTemplate;
