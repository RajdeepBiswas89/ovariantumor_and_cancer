
import React, { useState } from 'react';
import { ShieldCheck, Mail, Lock, Chrome, Building2, ArrowRight, Activity, Eye, EyeOff, Loader2 } from 'lucide-react';

interface AuthProps {
  onLogin: (userData: any) => void;
}

const Auth: React.FC<AuthProps> = ({ onLogin }) => {
  console.log('Auth component rendering');
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [microsoftLoading, setMicrosoftLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Simulate industry-grade auth latency
    setTimeout(() => {
      onLogin({ name: 'Dr. Sarah Mitchell', role: 'Chief Pathologist', institution: 'General Medical Center' });
      setLoading(false);
    }, 1500);
  };

  const handleSocialLogin = (provider: 'google' | 'microsoft') => {
    if (provider === 'google') setGoogleLoading(true);
    if (provider === 'microsoft') setMicrosoftLoading(true);

    // Simulate OAuth handshake
    setTimeout(() => {
      onLogin({ 
        name: 'Dr. Sarah Mitchell', 
        role: 'Chief Pathologist', 
        institution: provider === 'google' ? 'Google Health Cloud' : 'Microsoft Healthcare' 
      });
      if (provider === 'google') setGoogleLoading(false);
      if (provider === 'microsoft') setMicrosoftLoading(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden bg-slate-950 px-4">
      {/* Dynamic Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full opacity-20 surgical-grid pointer-events-none"></div>
      <div className="absolute -top-24 -left-24 w-96 h-96 bg-teal-600/20 rounded-full blur-[120px] animate-pulse"></div>
      <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-blue-600/20 rounded-full blur-[120px] animate-pulse delay-1000"></div>

      <div className="max-w-md w-full relative z-10">
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center bg-slate-900 p-4 rounded-3xl text-teal-500 shadow-2xl mb-6 ring-1 ring-white/10">
            <Activity size={40} />
          </div>
          <h1 className="text-4xl font-black text-white tracking-tighter mb-2">
            Ova<span className="text-teal-500">Scan</span> AI
          </h1>
          <p className="text-slate-400 font-medium tracking-tight">Institutional Access Gateway</p>
        </div>

        <div className="glass-morphism bg-white/5 border border-white/10 rounded-[2.5rem] p-10 shadow-3xl backdrop-blur-2xl">
          <div className="flex bg-slate-900/50 p-1 rounded-2xl mb-8 ring-1 ring-white/5">
            <button 
              onClick={() => setIsLogin(true)}
              disabled={loading || googleLoading || microsoftLoading}
              className={`flex-1 py-2.5 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${isLogin ? 'bg-teal-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'} disabled:opacity-50`}
            >
              Credentials
            </button>
            <button 
              onClick={() => setIsLogin(false)}
              disabled={loading || googleLoading || microsoftLoading}
              className={`flex-1 py-2.5 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${!isLogin ? 'bg-teal-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'} disabled:opacity-50`}
            >
              Enrollment
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-1.5">
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Professional Email</label>
              <div className="relative">
                <Mail size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
                <input 
                  type="email" 
                  required
                  disabled={loading || googleLoading || microsoftLoading}
                  className="w-full bg-slate-900/50 border border-white/10 rounded-2xl pl-12 pr-4 py-4 text-white focus:ring-2 focus:ring-teal-500 outline-none transition-all placeholder:text-slate-600 disabled:opacity-50"
                  placeholder="name@medical-center.org"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Security Token</label>
              <div className="relative">
                <Lock size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
                <input 
                  type={showPassword ? "text" : "password"} 
                  required
                  disabled={loading || googleLoading || microsoftLoading}
                  className="w-full bg-slate-900/50 border border-white/10 rounded-2xl pl-12 pr-12 py-4 text-white focus:ring-2 focus:ring-teal-500 outline-none transition-all placeholder:text-slate-600 disabled:opacity-50"
                  placeholder="••••••••••••"
                />
                <button 
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={loading || googleLoading || microsoftLoading}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 disabled:opacity-0"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <button 
              type="submit"
              disabled={loading || googleLoading || microsoftLoading}
              className="w-full bg-teal-600 hover:bg-teal-700 text-white py-4 rounded-2xl font-black uppercase tracking-widest text-sm shadow-xl shadow-teal-600/20 transition-all flex items-center justify-center gap-3 disabled:opacity-50 group"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Verify & Continue'}
              {!loading && <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />}
            </button>
          </form>

          <div className="mt-8 flex items-center gap-4">
            <div className="h-px flex-1 bg-white/10"></div>
            <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">Institutional SSO</span>
            <div className="h-px flex-1 bg-white/10"></div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-6">
            <button 
              onClick={() => handleSocialLogin('google')}
              disabled={loading || googleLoading || microsoftLoading}
              className="flex items-center justify-center gap-2 bg-white/5 border border-white/10 hover:bg-white/10 py-3 rounded-2xl transition-all disabled:opacity-50 relative overflow-hidden"
            >
              {googleLoading ? (
                <Loader2 size={18} className="text-teal-500 animate-spin" />
              ) : (
                <Chrome size={20} className="text-teal-500" />
              )}
              <span className="text-xs font-bold text-slate-300">{googleLoading ? 'Connecting...' : 'Google Health'}</span>
            </button>
            <button 
              onClick={() => handleSocialLogin('microsoft')}
              disabled={loading || googleLoading || microsoftLoading}
              className="flex items-center justify-center gap-2 bg-white/5 border border-white/10 hover:bg-white/10 py-3 rounded-2xl transition-all disabled:opacity-50 relative overflow-hidden"
            >
              {microsoftLoading ? (
                <Loader2 size={18} className="text-blue-500 animate-spin" />
              ) : (
                <Building2 size={20} className="text-blue-500" />
              )}
              <span className="text-xs font-bold text-slate-300">{microsoftLoading ? 'Authenticating...' : 'Microsoft MS'}</span>
            </button>
          </div>
        </div>

        <div className="mt-10 flex items-center justify-center gap-2 text-slate-500">
          <ShieldCheck size={16} className="text-teal-600" />
          <p className="text-[10px] font-bold uppercase tracking-widest">HIPAA & SOC2 TYPE II COMPLIANT ENVIRONMENT</p>
        </div>
      </div>
    </div>
  );
};

export default Auth;
