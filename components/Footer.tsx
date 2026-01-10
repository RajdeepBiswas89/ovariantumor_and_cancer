
import React from 'react';
import { Activity, Mail, Globe, ShieldCheck } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-slate-200 pt-20 pb-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-12 mb-16">
          <div className="col-span-2">
            <div className="flex items-center gap-2 mb-6">
              <div className="bg-teal-600 p-2 rounded-xl text-white">
                <Activity size={24} />
              </div>
              <span className="text-2xl font-bold text-slate-800">
                Ova<span className="text-teal-600">Scan</span> AI
              </span>
            </div>
            <p className="text-slate-500 max-w-sm mb-8">
              Pioneering the future of gynecological diagnostics through advanced deep learning and clinical research.
            </p>
            <div className="flex gap-4">
               <SocialLink icon={<Mail size={20} />} />
               <SocialLink icon={<Globe size={20} />} />
               <SocialLink icon={<ShieldCheck size={20} />} />
            </div>
          </div>
          
          <div>
            <h4 className="font-bold text-slate-900 mb-6">Technology</h4>
            <ul className="space-y-4 text-slate-500 text-sm">
              <li><a href="#" className="hover:text-teal-600">ResNet Hybrid Models</a></li>
              <li><a href="#" className="hover:text-teal-600">Clinical Data Pipelines</a></li>
              <li><a href="#" className="hover:text-teal-600">Inference Engine</a></li>
              <li><a href="#" className="hover:text-teal-600">Security & HIPAA</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-bold text-slate-900 mb-6">Company</h4>
            <ul className="space-y-4 text-slate-500 text-sm">
              <li><a href="#" className="hover:text-teal-600">About Us</a></li>
              <li><a href="#" className="hover:text-teal-600">Research Papers</a></li>
              <li><a href="#" className="hover:text-teal-600">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-teal-600">Contact Support</a></li>
            </ul>
          </div>
        </div>
        
        <div className="pt-10 border-t border-slate-100 flex flex-col md:flex-row justify-between items-center gap-6">
          <p className="text-slate-400 text-xs font-medium">
            © 2024 OvaScan AI Diagnostics. For Research & Educational Purposes Only.
          </p>
          <div className="flex gap-8">
            <a href="#" className="text-xs font-bold text-slate-400 hover:text-slate-900 uppercase tracking-widest">Privacy</a>
            <a href="#" className="text-xs font-bold text-slate-400 hover:text-slate-900 uppercase tracking-widest">Terms</a>
            <a href="#" className="text-xs font-bold text-slate-400 hover:text-slate-900 uppercase tracking-widest">Clinical Disclaimer</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

const SocialLink: React.FC<{ icon: React.ReactNode }> = ({ icon }) => (
  <a href="#" className="w-10 h-10 rounded-xl bg-slate-50 flex items-center justify-center text-slate-400 hover:bg-teal-50 hover:text-teal-600 transition-all border border-slate-100">
    {icon}
  </a>
);

export default Footer;
