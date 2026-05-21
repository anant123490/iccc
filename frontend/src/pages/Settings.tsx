import { useEffect, useState } from 'react';
import { Moon, Sun, Shield, Server } from 'lucide-react';
import { getSettings, saveSettings } from '../services/storage';
import type { AppSettings } from '../types';
import { useTheme } from '../hooks/useTheme';
import { loadModel } from '../services/inference';

export default function Settings() {
  const { darkMode, toggle } = useTheme();
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);

  useEffect(() => {
    getSettings().then(setSettings);
    loadModel().then(setModelLoaded);
  }, []);

  const update = async (patch: Partial<AppSettings>) => {
    if (!settings) return;
    const next = { ...settings, ...patch };
    await saveSettings(next);
    setSettings(next);
  };

  if (!settings) return null;

  return (
    <div className="space-y-4">
      <h2 className="font-bold text-lg">Settings</h2>

      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {darkMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            <span>Dark Mode</span>
          </div>
          <button
            onClick={toggle}
            className={`w-12 h-6 rounded-full transition-colors ${darkMode ? 'bg-medical-600' : 'bg-slate-300'}`}
          >
            <div className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${darkMode ? 'translate-x-6' : 'translate-x-0.5'}`} />
          </button>
        </div>

        <div>
          <label className="text-sm font-medium">ICDAS Mode</label>
          <select
            value={settings.icdasMode}
            onChange={(e) => update({ icdasMode: e.target.value as '0-4' | '0-6' })}
            className="w-full mt-1 px-4 py-2 rounded-lg border dark:bg-slate-700 dark:border-slate-600"
          >
            <option value="0-6">ICDAS 0–6 (7 classes)</option>
            <option value="0-4">ICDAS 0–4 (5 classes)</option>
          </select>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5" />
            <span>Encrypt Local Records</span>
          </div>
          <input
            type="checkbox"
            checked={settings.encryptionEnabled}
            onChange={(e) => update({ encryptionEnabled: e.target.checked })}
            className="w-5 h-5 accent-medical-600"
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Server className="w-5 h-5" />
            <span>Use Backend API</span>
          </div>
          <input
            type="checkbox"
            checked={settings.useBackend}
            onChange={(e) => update({ useBackend: e.target.checked })}
            className="w-5 h-5 accent-medical-600"
          />
        </div>

        {settings.useBackend && (
          <input
            type="url"
            value={settings.backendUrl}
            onChange={(e) => update({ backendUrl: e.target.value })}
            placeholder="Backend URL"
            className="w-full px-4 py-2 rounded-lg border dark:bg-slate-700"
          />
        )}
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Model Status</h3>
        <p className="text-sm">
          Offline TF.js model:{' '}
          <span className={modelLoaded ? 'text-green-600' : 'text-amber-600'}>
            {modelLoaded ? 'Loaded' : 'Demo mode (train & export model)'}
          </span>
        </p>
      </div>

      <div className="card text-sm text-slate-500">
        <p>All patient data is stored locally on this device. No cloud upload by default.</p>
      </div>
    </div>
  );
}
