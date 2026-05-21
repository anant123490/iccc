export interface ScanResult {
  id: string;
  patientId?: string;
  timestamp: string;
  icdasGrade: number;
  confidence: number;
  label: string;
  action: string;
  description: string;
  finding: string;
  recommendation: string;
  urgency: string;
  originalImage: string;
  heatmapImage?: string;
  overlayImage?: string;
  inferenceMs?: number;
  isDemo?: boolean;
}

export interface AppSettings {
  darkMode: boolean;
  icdasMode: '0-4' | '0-6';
  consentGiven: boolean;
  encryptionEnabled: boolean;
  useBackend: boolean;
  backendUrl: string;
}

export const DEFAULT_SETTINGS: AppSettings = {
  darkMode: false,
  icdasMode: '0-6',
  consentGiven: false,
  encryptionEnabled: true,
  useBackend: false,
  backendUrl: 'http://localhost:8000',
};

export const DISCLAIMER =
  'This tool is for clinical decision support and is not a substitute for professional diagnosis.';

export const AI_RESULT_NOTE = 'AI result is not final diagnosis.';

export interface IcdasAction {
  label: string;
  action: string;
  description: string;
  finding: string;
  recommendation: string;
  urgency: string;
}

export const ICDAS_ACTIONS: Record<number, IcdasAction> = {
  0: {
    label: 'Sound',
    action: 'Monitor',
    description: 'No evidence of caries.',
    finding: 'Sound tooth surface',
    recommendation: 'Continue routine monitoring and preventive care',
    urgency: 'low',
  },
  1: {
    label: 'Initial lesion',
    action: 'Monitor',
    description: 'First visual change in enamel.',
    finding: 'First visual change in enamel',
    recommendation: 'Monitor + reinforce oral hygiene',
    urgency: 'low',
  },
  2: {
    label: 'Distinct visual change',
    action: 'Fluoride treatment',
    description: 'Consider fluoride varnish.',
    finding: 'Distinct visual change in enamel',
    recommendation: 'Dentist review + preventive fluoride treatment',
    urgency: 'medium',
  },
  3: {
    label: 'Localized breakdown',
    action: 'Restoration needed',
    description: 'Enamel breakdown detected.',
    finding: 'Localized enamel breakdown',
    recommendation: 'Dentist review + restorative assessment',
    urgency: 'high',
  },
  4: {
    label: 'Underlying dentin',
    action: 'Restoration needed',
    description: 'Dentin shadow visible.',
    finding: 'Underlying dentin shadow',
    recommendation: 'Dentist review + restoration needed',
    urgency: 'high',
  },
  5: {
    label: 'Distinct cavity',
    action: 'Restoration needed',
    description: 'Cavity with visible dentin.',
    finding: 'Distinct cavity with visible dentin',
    recommendation: 'Prompt restorative treatment by dentist',
    urgency: 'critical',
  },
  6: {
    label: 'Extensive cavity',
    action: 'Restoration needed',
    description: 'Extensive cavity — urgent care.',
    finding: 'Extensive distinct cavity with dentin involvement',
    recommendation: 'Urgent dentist review + restoration',
    urgency: 'critical',
  },
};

/** Resolve finding/recommendation for scans saved before these fields existed. */
export function resolveClinicalText(result: ScanResult): Pick<ScanResult, 'finding' | 'recommendation'> {
  const action = ICDAS_ACTIONS[result.icdasGrade] ?? ICDAS_ACTIONS[0];
  return {
    finding: result.finding || action.finding,
    recommendation: result.recommendation || action.recommendation,
  };
}
