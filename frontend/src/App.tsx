import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { registerSW } from 'virtual:pwa-register';
import Layout from './components/Layout';
import Consent from './pages/Consent';
import Home from './pages/Home';
import Scan from './pages/Scan';
import Results from './pages/Results';
import History from './pages/History';
import Settings from './pages/Settings';
import { getSettings, saveSettings } from './services/storage';
import { loadModel } from './services/inference';

const updateSW = registerSW({ onNeedRefresh: () => {}, onOfflineReady: () => {} });

export default function App() {
  const [consent, setConsent] = useState<boolean | null>(null);

  useEffect(() => {
    getSettings().then((s) => setConsent(s.consentGiven));
    loadModel();
    updateSW(true);
  }, []);

  const acceptConsent = async () => {
    const s = await getSettings();
    s.consentGiven = true;
    await saveSettings(s);
    setConsent(true);
  };

  if (consent === null) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-medical-600">Loading...</div>
      </div>
    );
  }

  if (!consent) {
    return <Consent onAccept={acceptConsent} />;
  }

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/scan" element={<Scan />} />
          <Route path="/results" element={<Results />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
