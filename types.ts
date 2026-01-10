
export enum DiagnosisClass {
  NOT_INFECTED = 'Not Infected (Healthy)',
  INFECTED = 'Infected (Inflammation/Infection)',
  OVARIAN_TUMOR = 'Ovarian Tumor (Benign)',
  OVARIAN_CANCER = 'Ovarian Cancer (Malignant)'
}

export interface AnalysisResult {
  diagnosis: DiagnosisClass;
  confidence: number;
  findings: string;
  recommendations: string[];
  timestamp: string;
  patientId: string;
  patientName: string;
}

export interface PatientInfo {
  name: string;
  age: string;
  history: string;
  gender: string;
}
