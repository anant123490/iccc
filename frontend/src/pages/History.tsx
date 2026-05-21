import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Trash2, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { getScans, deleteScan } from '../services/storage';
import type { ScanResult } from '../types';

export default function History() {
  const [scans, setScans] = useState<ScanResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getScans().then((s) => {
      setScans(s);
      setLoading(false);
    });
  }, []);

  const handleDelete = async (id: string) => {
    await deleteScan(id);
    setScans((prev) => prev.filter((s) => s.id !== id));
  };

  const getTrend = (patientScans: ScanResult[]) => {
    if (patientScans.length < 2) return null;
    const sorted = [...patientScans].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    const delta = sorted[sorted.length - 1].icdasGrade - sorted[0].icdasGrade;
    if (delta > 0) return <TrendingUp className="w-4 h-4 text-red-500" />;
    if (delta < 0) return <TrendingDown className="w-4 h-4 text-green-500" />;
    return <Minus className="w-4 h-4 text-slate-400" />;
  };

  // Group by patient
  const grouped = scans.reduce<Record<string, ScanResult[]>>((acc, s) => {
    const key = s.patientId || 'anonymous';
    (acc[key] ??= []).push(s);
    return acc;
  }, {});

  if (loading) return <p className="text-center text-slate-500">Loading history...</p>;

  if (scans.length === 0) {
    return (
      <div className="card text-center">
        <p className="text-slate-500">No scans yet.</p>
        <Link to="/scan" className="btn-primary inline-block mt-4">Start Scan</Link>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="font-bold text-lg">Scan History</h2>
      {Object.entries(grouped).map(([patient, patientScans]) => (
        <div key={patient} className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
            Patient: {patient}
            {getTrend(patientScans)}
          </div>
          {patientScans.map((scan) => (
            <div key={scan.id} className="card flex gap-4 p-4">
              <img
                src={scan.originalImage}
                alt=""
                className="w-20 h-20 rounded-lg object-cover"
              />
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-2xl font-bold text-medical-600">{scan.icdasGrade}</span>
                    <span className="text-sm text-slate-500 ml-2">{scan.confidence}%</span>
                  </div>
                  <button
                    onClick={() => handleDelete(scan.id)}
                    className="text-slate-400 hover:text-red-500 p-1"
                    aria-label="Delete scan"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-sm font-medium truncate">{scan.label}</p>
                <p className="text-xs text-slate-500">
                  {new Date(scan.timestamp).toLocaleString()}
                </p>
                <p className="text-xs text-medical-600">{scan.action}</p>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
