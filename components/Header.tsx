
import React, { useState } from 'react';
import { Activity, HeartPulse, LayoutDashboard, BrainCircuit, Microscope, LogOut, ChevronDown, User, Settings } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: any) => void;
  user: any;
  onLogout: () => void;
}

const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, user, onLogout }) => {
  const [showProfileMenu, setShowProfileMenu] = useState(false);

  return (
    <header className="sticky top-0 z-50 glass-morphism border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-24">
          <div 
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => setActiveTab('home')}
          >
            <div className="bg-slate-900 p-2.5 rounded-2xl text-teal-500 group-hover:scale-105 transition-transform shadow-xl">
              <Activity size={28} />
            </div>
            <div>
              <span className="text-2xl font-black tracking-tighter text-slate-900 block leading-none">
                Ova<span className="text-teal-600">Scan</span>
              </span>
              <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mt-1 block">Clinical AI Suite</span>
            </div>
          </div>
          
          <nav className="hidden lg:flex items-center gap-10">
            <NavItem 
              active={activeTab === 'dashboard'} 
              onClick={() => setActiveTab('dashboard')}
              icon={<LayoutDashboard size={18} />}
              label="Dashboard"
            />
            <NavItem 
              active={activeTab === 'diagnose'} 
              onClick={() => setActiveTab('diagnose')}
              icon={<BrainCircuit size={18} />}
              label="Diagnostic Tool"
            />
            <NavItem 
              active={activeTab === 'assistant'} 
              onClick={() => setActiveTab('assistant')}
              icon={<Microscope size={18} />}
              label="Assistant"
            />
             <NavItem 
              active={activeTab === 'solutions'} 
              onClick={() => setActiveTab('solutions')}
              icon={<HeartPulse size={18} />}
              label="Solutions"
            />
          </nav>

          <div className="flex items-center gap-4">
            <div className="relative">
              <button 
                onClick={() => setShowProfileMenu(!showProfileMenu)}
                className="flex items-center gap-3 pr-4 border-r border-slate-200 hover:bg-slate-50 transition-all rounded-2xl py-2 group"
              >
                <div className="w-10 h-10 rounded-full bg-slate-900 flex items-center justify-center font-bold text-teal-500 text-xs shadow-lg group-hover:scale-105 transition-transform">
                  {user.name.split(' ').map((n: string) => n[0]).join('')}
                </div>
                <div className="text-left hidden sm:block">
                  <p className="text-xs font-black text-slate-900 leading-none">{user.name}</p>
                  <p className="text-[10px] text-teal-600 font-bold uppercase mt-1">{user.role}</p>
                </div>
                <ChevronDown size={14} className={`text-slate-400 transition-transform ${showProfileMenu ? 'rotate-180' : ''}`} />
              </button>

              {showProfileMenu && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowProfileMenu(false)}></div>
                  <div className="absolute right-0 mt-4 w-64 bg-white border border-slate-200 rounded-3xl shadow-2xl py-3 z-50 animate-fade-in">
                    <div className="px-6 py-4 border-b border-slate-100 mb-2">
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Institutional Entity</p>
                      <p className="text-sm font-bold text-slate-900">{user.institution}</p>
                    </div>
                    <MenuButton icon={<User size={16} />} label="Security Profile" />
                    <MenuButton icon={<Settings size={16} />} label="System Config" />
                    <div className="h-px bg-slate-100 my-2 mx-4"></div>
                    <button 
                      onClick={onLogout}
                      className="w-full flex items-center gap-3 px-6 py-3 text-red-600 hover:bg-red-50 font-bold text-sm transition-all"
                    >
                      <LogOut size={16} />
                      Revoke Credentials
                    </button>
                  </div>
                </>
              )}
            </div>
            <button 
              onClick={() => setActiveTab('diagnose')}
              className="bg-teal-600 text-white px-6 py-3 rounded-xl font-bold text-sm hover:bg-teal-700 transition-all shadow-lg hover:shadow-teal-600/20 active:scale-95"
            >
              Execute Scan
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

const NavItem: React.FC<{ active: boolean; onClick: () => void; icon: React.ReactNode; label: string }> = ({ active, onClick, icon, label }) => (
  <button 
    onClick={onClick}
    className={`flex items-center gap-2 py-2 px-1 transition-all text-sm font-bold uppercase tracking-widest ${
      active ? 'text-teal-600 border-b-2 border-teal-600' : 'text-slate-400 hover:text-slate-900 border-b-2 border-transparent'
    }`}
  >
    {icon}
    {label}
  </button>
);

const MenuButton: React.FC<{ icon: React.ReactNode; label: string }> = ({ icon, label }) => (
  <button className="w-full flex items-center gap-3 px-6 py-3 text-slate-600 hover:bg-slate-50 font-bold text-sm transition-all">
    {icon}
    {label}
  </button>
);

export default Header;
