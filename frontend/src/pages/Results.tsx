import { useLocation, Link } from 'react-router-dom';
import type { ScanResult } from '../types';
import AiSuggestionCard from '../components/AiSuggestionCard';

export default function Results() {
  const location = useLocation();
  const result = location.state?.result as ScanResult | undefined;

  if (!result) {
    return (
      <div className="card text-center">
        <p>No scan results. Start a new scan.</p>
        <Link to="/scan" className="btn-primary inline-block mt-4">New Scan</Link>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <AiSuggestionCard result={result} />

      <div className="grid grid-cols-1 gap-4">
        <div className="card p-0 overflow-hidden">
          <p className="px-4 py-2 text-sm font-medium border-b dark:border-slate-700">Original Image</p>
          <img src={result.originalImage} alt="Original" className="w-full" />
        </div>
        {result.overlayImage && (
          <div className="card p-0 overflow-hidden">
            <p className="px-4 py-2 text-sm font-medium border-b dark:border-slate-700">Grad-CAM Heatmap Overlay</p>
            <img src={result.overlayImage} alt="Heatmap overlay" className="w-full" />
          </div>
        )}
      </div>

      <Link to="/scan" className="btn-primary block text-center">New Scan</Link>
      <Link to="/history" className="block text-center text-medical-600 text-sm font-medium">
        View in History
      </Link>
    </div>
  );
}
