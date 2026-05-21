import type { ScanResult } from '../types';
import { AI_RESULT_NOTE, resolveClinicalText } from '../types';
import clsx from 'clsx';

const urgencyColors: Record<string, string> = {
  low: 'border-green-200 dark:border-green-800',
  medium: 'border-yellow-200 dark:border-yellow-800',
  high: 'border-orange-200 dark:border-orange-800',
  critical: 'border-red-200 dark:border-red-800',
};

interface Props {
  result: ScanResult;
  showDemoBadge?: boolean;
}

function SuggestionLine({ label, value }: { label: string; value: string }) {
  return (
    <p className="text-sm leading-relaxed">
      <span className="font-semibold text-slate-800 dark:text-slate-100">{label}: </span>
      <span className="text-slate-700 dark:text-slate-200">{value}</span>
    </p>
  );
}

export default function AiSuggestionCard({ result, showDemoBadge = true }: Props) {
  const { finding, recommendation } = resolveClinicalText(result);
  const confidence =
    result.confidence % 1 === 0 ? String(result.confidence) : result.confidence.toFixed(1);

  return (
    <div
      className={clsx(
        'card border-l-4 space-y-3',
        urgencyColors[result.urgency] ?? urgencyColors.low
      )}
    >
      {showDemoBadge && result.isDemo && (
        <p className="text-xs font-medium text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/50 px-2 py-1 rounded">
          Demo mode — connect a trained model or backend API for clinical inference
        </p>
      )}
      <SuggestionLine label="AI Suggestion" value={`ICDAS Grade ${result.icdasGrade}`} />
      <SuggestionLine label="Confidence" value={`${confidence}%`} />
      <SuggestionLine label="Finding" value={finding} />
      <SuggestionLine label="Recommendation" value={recommendation} />
      <SuggestionLine label="Note" value={AI_RESULT_NOTE} />
      {result.inferenceMs != null && (
        <p className="text-xs text-slate-400 pt-1">Processed in {result.inferenceMs}ms</p>
      )}
    </div>
  );
}
