import { Link, useLocation } from 'react-router-dom';
import { Home, Camera, History, Settings, Activity } from 'lucide-react';
import { DISCLAIMER } from '../types';
import clsx from 'clsx';

const nav = [
  { to: '/', icon: Home, label: 'Home' },
  { to: '/scan', icon: Camera, label: 'Scan' },
  { to: '/history', icon: History, label: 'History' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col pb-20">
      <header className="bg-medical-600 dark:bg-medical-900 text-white px-4 py-4 shadow-lg">
        <div className="max-w-lg mx-auto flex items-center gap-3">
          <Activity className="w-8 h-8" />
          <div>
            <h1 className="text-lg font-bold">ICDAS Dental Scan</h1>
            <p className="text-xs text-medical-100 opacity-90">Offline Edge AI</p>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-lg mx-auto w-full px-4 py-6">{children}</main>

      <p className="text-center text-xs text-slate-500 dark:text-slate-400 px-6 pb-24 max-w-lg mx-auto">
        {DISCLAIMER}
      </p>

      <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 safe-area-pb">
        <div className="max-w-lg mx-auto flex justify-around py-2">
          {nav.map(({ to, icon: Icon, label }) => (
            <Link
              key={to}
              to={to}
              className={clsx(
                'flex flex-col items-center gap-1 px-4 py-2 rounded-lg text-xs font-medium transition-colors',
                location.pathname === to
                  ? 'text-medical-600 dark:text-medical-400'
                  : 'text-slate-500 hover:text-medical-500'
              )}
            >
              <Icon className="w-6 h-6" />
              {label}
            </Link>
          ))}
        </div>
      </nav>
    </div>
  );
}
