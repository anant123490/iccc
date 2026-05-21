import { useRef, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, Loader2 } from 'lucide-react';
import { runInference } from '../services/inference';
import { saveScan } from '../services/storage';

export default function Scan() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [patientId, setPatientId] = useState('');
  const [error, setError] = useState('');

  const startCamera = async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      setStream(media);
      if (videoRef.current) {
        videoRef.current.srcObject = media;
        await videoRef.current.play();
      }
      setPreview(null);
    } catch {
      setError('Camera access denied. Use gallery upload instead.');
    }
  };

  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const v = videoRef.current;
    const c = canvasRef.current;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    setPreview(c.toDataURL('image/jpeg', 0.9));
    stopCamera();
  };

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(file);
    stopCamera();
  };

  const analyze = useCallback(async () => {
    if (!preview) return;
    setLoading(true);
    setError('');
    try {
      const img = new Image();
      await new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = rej;
        img.src = preview;
      });
      const result = await runInference(img, patientId || undefined);
      await saveScan(result);
      navigate('/results', { state: { result } });
    } catch (err) {
      setError('Analysis failed. Please try another image.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [preview, patientId, navigate]);

  return (
    <div className="space-y-4">
      <div className="card p-0 overflow-hidden">
        {preview ? (
          <img src={preview} alt="Preview" className="w-full aspect-[4/3] object-cover" />
        ) : stream ? (
          <video ref={videoRef} className="w-full aspect-[4/3] object-cover" playsInline muted />
        ) : (
          <div className="aspect-[4/3] bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
            <Camera className="w-16 h-16 text-slate-400" />
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      <input
        type="text"
        placeholder="Patient ID (optional)"
        value={patientId}
        onChange={(e) => setPatientId(e.target.value)}
        className="w-full px-4 py-3 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800"
      />

      <div className="grid grid-cols-2 gap-3">
        {!stream && !preview && (
          <button onClick={startCamera} className="btn-primary flex items-center justify-center gap-2">
            <Camera className="w-5 h-5" /> Camera
          </button>
        )}
        {stream && (
          <button onClick={capturePhoto} className="btn-primary col-span-2">
            Capture Photo
          </button>
        )}
        <button
          onClick={() => fileRef.current?.click()}
          className="flex items-center justify-center gap-2 py-3 px-4 rounded-xl border-2 border-medical-600 text-medical-600 font-medium"
        >
          <Upload className="w-5 h-5" /> Gallery
        </button>
        <input ref={fileRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={handleFile} />
      </div>

      {preview && (
        <button onClick={analyze} disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
          {loading ? <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing...</> : 'Analyze Image'}
        </button>
      )}

      {error && <p className="text-red-500 text-sm text-center">{error}</p>}
    </div>
  );
}
