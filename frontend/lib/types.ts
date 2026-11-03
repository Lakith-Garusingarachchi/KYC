export type CheckStatus = "pass" | "warn" | "fail";

export interface Check {
  id: string;
  category: "document" | "address" | "liveness" | "face" | string;
  label: string;
  status: CheckStatus;
  detail: string;
}

export interface User {
  id: string;
  email: string;
  full_name: string;
  auth_provider: string;
}

export type SubmissionStatus = "draft" | "documents_uploaded" | "processing" | "completed" | "failed";
export type Decision = "approved" | "manual_review" | "rejected";

export interface FaceMatch {
  score?: number | null;
  center_score?: number;
  verdict: "match" | "uncertain" | "mismatch" | "no_id_face";
}

export interface Submission {
  id: string;
  status: SubmissionStatus;
  decision: Decision | null;
  score: number | null;
  pass_score: number | null;
  blockers: string[];
  progress: string | null;
  id_type: "nic" | "passport" | null;
  address_input: string | null;
  uploaded: string[];
  quality: Record<string, { issues?: string[] }>;
  extracted: {
    identity?: Record<string, string | null>;
    bill?: Record<string, string | null>;
    identity_error?: string;
    bill_error?: string;
  };
  face: {
    liveness_pass?: boolean;
    pose_checks?: Record<string, boolean>;
    face_match?: FaceMatch;
    issues?: string[];
    ai_review?: { real_live_capture?: boolean; spoof_signs?: string[]; notes?: string } | null;
  };
  checks: Check[];
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface Challenge {
  challenge_id: string;
  steps: PoseStep[];
  expires_in: number;
}

export type PoseStep = "center" | "left" | "right" | "up" | "down";

export interface LivenessResult {
  accepted: boolean;
  liveness_pass: boolean;
  attempts_left: number;
  issues: string[];
  face_match: FaceMatch;
  submission_status: SubmissionStatus;
}
