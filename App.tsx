
import React, { useState } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import Hero from './components/Hero';
import DiagnosticTool from './components/DiagnosticTool';
import Solutions from './components/Solutions';
import ResearchSection from './components/ResearchSection';
import Dashboard from './components/Dashboard';
import ClinicalAssistant from './components/ClinicalAssistant';
import Auth from './components/Auth';

const App: React.FC = () => {
  console.log('App component rendering');
  const [activeTab, setActiveTab] = useState<'home' | 'dashboard' | 'diagnose' | 'assistant' | 'solutions'>('home');
  const [user, setUser] = useState<any>(null);

  const handleLogin = (userData: any) => {
    setUser(userData);
    setActiveTab('dashboard'); // Default to dashboard after high-end login
  };

  const handleLogout = () => {
    setUser(null);
    setActiveTab('home');
  };

  if (!user) {
    return <Auth onLogin={handleLogin} />;
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return (
          <>
            <Hero onStart={() => setActiveTab('diagnose')} />
            <ResearchSection />
            <Solutions />
          </>
        );
      case 'dashboard':
        return <Dashboard />;
      case 'diagnose':
        return <DiagnosticTool />;
      case 'assistant':
        return <ClinicalAssistant />;
      case 'solutions':
        return <Solutions />;
      default:
        return <Hero onStart={() => setActiveTab('diagnose')} />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} user={user} onLogout={handleLogout} />
      <main className="flex-grow">
        {renderContent()}
      </main>
      <Footer />
    </div>
  );
};

export default App;
