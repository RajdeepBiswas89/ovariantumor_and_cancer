
import React from 'react';
import { Activity, Users, ShieldAlert, Cpu, TrendingUp, Globe, Clock } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

const velocityData = [
  { name: '08:00', value: 45 },
  { name: '10:00', value: 89 },
  { name: '12:00', value: 124 },
  { name: '14:00', value: 78 },
  { name: '16:00', value: 156 },
  { name: '18:00', value: 92 },
];

const diagnosticTrends = [
  { val: 65 }, { val: 72 }, { val: 68 }, { val: 85 }, { val: 82 }, { val: 94 }, { val: 105 }
];
const malignancyTrends = [
  { val: 5.1 }, { val: 4.9 }, { val: 4.7 }, { val: 4.8 }, { val: 4.5 }, { val: 4.3 }, { val: 4.2 }
];
const institutionTrends = [
  { val: 148 }, { val: 149 }, { val: 150 }, { val: 152 }, { val: 153 }, { val: 155 }, { val: 156 }
];
const inferenceTrends = [
  { val: 320 }, { val: 305 }, { val: 290 }, { val: 275 }, { val: 260 }, { val: 250 }, { val: 240 }
];

const Dashboard: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-12">
      <div className="flex justify-between items-end mb-10">
        <div>
          <h2 className="text-4xl font-black text-slate-900 tracking-tight">Clinical Command</h2>
          <p className="text-slate-500 font-medium">Institutional Diagnostic Telemetry</p>
        </div>
        <div className="flex gap-3">
          <div className="px-4 py-2 bg-white border border-slate-200 rounded-xl flex items-center gap-2 shadow-sm">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-xs font-bold text-slate-600 uppercase">ResNet Cluster Active</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard 
          icon={<Activity className="text-teal-600" />} 
          label="Total Diagnostics" 
          value="12,842" 
          delta="+14%" 
          trendData={diagnosticTrends}
        />
        <StatCard 
          icon={<ShieldAlert className="text-red-600" />} 
          label="Malignancy Rate" 
          value="4.2%" 
          delta="-0.8%" 
          trendData={malignancyTrends}
          inverse
        />
        <StatCard 
          icon={<Users className="text-blue-600" />} 
          label="Active Institutions" 
          value="156" 
          delta="+4" 
          trendData={institutionTrends}
        />
        <StatCard 
          icon={<Cpu className="text-purple-600" />} 
          label="Avg. Inference" 
          value="240ms" 
          delta="Optimal" 
          trendData={inferenceTrends}
        />
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white rounded-3xl p-8 border border-slate-200 shadow-xl shadow-slate-200/50">
          <div className="flex justify-between items-center mb-8">
            <h3 className="text-xl font-bold flex items-center gap-2">
              <TrendingUp size={20} className="text-teal-600" />
              Scan Velocity (24h)
            </h3>
            <select className="bg-slate-50 border-none text-xs font-bold text-slate-500 uppercase rounded-lg px-3 py-1 outline-none">
              <option>Real-time</option>
              <option>Weekly</option>
            </select>
          </div>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={velocityData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0d9488" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#0d9488" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 10, fontWeight: 700, fill: '#94a3b8'}} />
                <YAxis hide />
                <Tooltip 
                   contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}}
                   itemStyle={{fontSize: '12px', fontWeight: 700}}
                />
                <Area type="monotone" dataKey="value" stroke="#0d9488" strokeWidth={3} fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-900 rounded-3xl p-8 text-white flex flex-col">
          <h3 className="text-xl font-bold mb-8 flex items-center gap-2">
            <Globe size={20} className="text-teal-400" />
            Global Reach
          </h3>
          <div className="flex-grow space-y-6">
            <RegionItem name="North America" count="4,201" progress={85} color="bg-teal-500" />
            <RegionItem name="Europe (EU-Central)" count="3,892" progress={72} color="bg-blue-500" />
            <RegionItem name="Asia Pacific" count="2,110" progress={45} color="bg-purple-500" />
            <RegionItem name="Other Regions" count="2,639" progress={55} color="bg-slate-500" />
          </div>
          <div className="mt-8 pt-6 border-t border-slate-800">
            <div className="flex items-center gap-3">
              <Clock size={16} className="text-slate-500" />
              <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">System Sync: 12s ago</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard: React.FC<{ 
  icon: React.ReactNode; 
  label: string; 
  value: string; 
  delta: string; 
  trendData: any[];
  inverse?: boolean;
}> = ({ icon, label, value, delta, trendData, inverse }) => {
  const isPositive = delta.startsWith('+');
  // For malignancy rate, negative delta is actually "good" (green)
  const isGood = inverse ? !isPositive : isPositive;
  const trendColor = isGood ? '#10b981' : '#3b82f6';

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex justify-between items-start mb-4">
          <div className="p-2 bg-slate-50 rounded-lg">{icon}</div>
          <div className="flex flex-col items-end">
             <span className={`text-[10px] font-black px-2 py-1 rounded-full ${isGood ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}>
              {delta}
            </span>
          </div>
        </div>
        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">{label}</p>
        <div className="flex items-end justify-between gap-2">
          <p className="text-2xl font-black text-slate-900 leading-none">{value}</p>
          <div className="h-8 w-16 mb-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData}>
                <Line 
                  type="monotone" 
                  dataKey="val" 
                  stroke={trendColor} 
                  strokeWidth={2} 
                  dot={false} 
                  isAnimationActive={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <p className="text-[8px] font-bold text-slate-300 uppercase tracking-wider mt-2">7D Trajectory vs. Benchmark</p>
      </div>
    </div>
  );
};

const RegionItem: React.FC<{ name: string; count: string; progress: number; color: string }> = ({ name, count, progress, color }) => (
  <div className="space-y-2">
    <div className="flex justify-between text-xs font-bold">
      <span className="text-slate-400 uppercase tracking-wider">{name}</span>
      <span className="text-white">{count}</span>
    </div>
    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
      <div className={`h-full ${color}`} style={{ width: `${progress}%` }}></div>
    </div>
  </div>
);

export default Dashboard;
