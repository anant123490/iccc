import { Link } from 'react-router-dom';
import { Camera, Shield, Zap, Eye, WifiOff } from 'lucide-react';

export default function Home() {
  return (
    <div className="space-y-6">
      <section className="card text-center">
        <h2 className="text-2xl font-bold text-medical-700 dark:text-medical-400 mb-2">
          Dental Caries Detection
        </h2>
        <p className="text-slate-600 dark:text-slate-300 text-sm">
          AI-powered ICDAS classification from intraoral photos. Works fully offline after install.
        </p>
        <Link to="/scan" className="btn-primary inline-flex items-center gap-2 mt-6 w-full justify-center">
          <Camera className="w-5 h-5" />
          Start New Scan
        </Link>
      </section>

      <div className="grid grid-cols-2 gap-4">
        {[
          { icon: WifiOff, title: 'Offline', desc: 'No internet needed' },
          { icon: Zap, title: '<1s', desc: 'Edge inference' },
          { icon: Eye, title: 'Grad-CAM', desc: 'Explainable AI' },
          { icon: Shield, title: 'Private', desc: 'Local storage only' },
        ].map(({ icon: Icon, title, desc }) => (
          <div key={title} className="card p-4 text-center">
            <Icon className="w-8 h-8 mx-auto text-medical-600 mb-2" />
            <p className="font-semibold text-sm">{title}</p>
            <p className="text-xs text-slate-500">{desc}</p>
          </div>
        ))}
      </div>

      <section className="card">
        <h3 className="font-semibold mb-3">ICDAS Scale (0–6)</h3>
        <ul className="text-sm space-y-2 text-slate-600 dark:text-slate-300">
          <li><span className="font-medium text-green-600">0</span> — Sound tooth</li>
          <li><span className="font-medium text-lime-600">1–2</span> — Monitor / Fluoride</li>
          <li><span className="font-medium text-amber-600">3–4</span> — Restoration needed</li>
          <li><span className="font-medium text-red-600">5–6</span> — Urgent restoration</li>
        </ul>
      </section>
    </div>
  );
}
