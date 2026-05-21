import { Shield } from 'lucide-react';
import { DISCLAIMER } from '../types';

interface Props {
  onAccept: () => void;
}

export default function Consent({ onAccept }: Props) {
  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-slate-50 dark:bg-slate-900">
      <div className="card max-w-md w-full space-y-6">
        <div className="text-center">
          <Shield className="w-16 h-16 mx-auto text-medical-600 mb-4" />
          <h1 className="text-xl font-bold">Consent & Privacy</h1>
        </div>

        <div className="text-sm space-y-3 text-slate-600 dark:text-slate-300">
          <p>Before using ICDAS Dental Scan, please acknowledge:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li>{DISCLAIMER}</li>
            <li>All images and results are stored <strong>locally on your device</strong> only.</li>
            <li>No data is uploaded to the cloud unless you explicitly enable the backend API.</li>
            <li>You are responsible for obtaining patient consent where required by law.</li>
            <li>AI predictions may be incorrect — always verify with clinical examination.</li>
          </ul>
        </div>

        <button onClick={onAccept} className="btn-primary w-full">
          I Understand & Accept
        </button>
      </div>
    </div>
  );
}
