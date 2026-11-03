// Head pose from MediaPipe face landmarks, using the same ratios the backend
// computes from YuNet landmarks (backend/app/services/face.py):
//   yaw   = (nose.x - eyeMid.x) / eyeDistance   > 0 when the person turns to THEIR left
//   pitch = (nose.y - eyeMid.y) / (mouthMid.y - eyeMid.y)   smaller when looking up
// Coordinates are from the unmirrored camera frame.
import type { PoseStep } from "./types";

export interface Point {
  x: number;
  y: number;
}

export interface Pose {
  yaw: number;
  pitch: number;
  faceWidth: number; // fraction of frame width
  centerX: number;
  centerY: number;
}

// MediaPipe Face Mesh landmark indices
const NOSE_TIP = 1;
const EYE_A = [33, 133];
const EYE_B = [362, 263];
const MOUTH = [61, 291];
const FACE_EDGES = [234, 454];

const mid = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

export function poseFromLandmarks(lm: Point[], frameW: number, frameH: number): Pose {
  // Convert normalized coords to pixels so x and y share a scale.
  const p = (i: number): Point => ({ x: lm[i].x * frameW, y: lm[i].y * frameH });
  const eyeA = mid(p(EYE_A[0]), p(EYE_A[1]));
  const eyeB = mid(p(EYE_B[0]), p(EYE_B[1]));
  const eyeMid = mid(eyeA, eyeB);
  const mouthMid = mid(p(MOUTH[0]), p(MOUTH[1]));
  const nose = p(NOSE_TIP);
  const eyeDist = Math.hypot(eyeB.x - eyeA.x, eyeB.y - eyeA.y) || 1;
  const faceH = mouthMid.y - eyeMid.y || 1;
  const left = p(FACE_EDGES[0]);
  const right = p(FACE_EDGES[1]);
  return {
    yaw: (nose.x - eyeMid.x) / eyeDist,
    pitch: (nose.y - eyeMid.y) / faceH,
    faceWidth: Math.abs(right.x - left.x) / frameW,
    centerX: (left.x + right.x) / 2 / frameW,
    centerY: eyeMid.y / frameH,
  };
}

// The browser asks for a clearer movement than the server requires (0.10 / 0.05),
// so frames that pass here pass there too.
export const CLIENT_YAW_DELTA = 0.2;
export const CLIENT_PITCH_DELTA = 0.09;
const CENTER_MAX_YAW = 0.12;

export function stepReached(step: PoseStep, pose: Pose, base: Pose | null): boolean {
  if (step === "center") return Math.abs(pose.yaw) < CENTER_MAX_YAW;
  if (!base) return false;
  switch (step) {
    case "left":
      return pose.yaw - base.yaw >= CLIENT_YAW_DELTA;
    case "right":
      return base.yaw - pose.yaw >= CLIENT_YAW_DELTA;
    case "up":
      return base.pitch - pose.pitch >= CLIENT_PITCH_DELTA;
    case "down":
      return pose.pitch - base.pitch >= CLIENT_PITCH_DELTA;
  }
}

/** 0..1 progress toward the required movement, for the on-screen meter. */
export function stepProgress(step: PoseStep, pose: Pose, base: Pose | null): number {
  if (step === "center") return Math.max(0, Math.min(1, 1 - Math.abs(pose.yaw) / 0.4));
  if (!base) return 0;
  const v =
    step === "left" ? (pose.yaw - base.yaw) / CLIENT_YAW_DELTA
    : step === "right" ? (base.yaw - pose.yaw) / CLIENT_YAW_DELTA
    : step === "up" ? (base.pitch - pose.pitch) / CLIENT_PITCH_DELTA
    : (pose.pitch - base.pitch) / CLIENT_PITCH_DELTA;
  return Math.max(0, Math.min(1, v));
}

export const STEP_TEXT: Record<PoseStep, { title: string; say: string }> = {
  center: { title: "Look straight at the camera", say: "Look straight at the camera and hold still." },
  left: { title: "Slowly turn your head to your left", say: "Now slowly turn your head to your left." },
  right: { title: "Slowly turn your head to your right", say: "Now slowly turn your head to your right." },
  up: { title: "Tilt your head up a little", say: "Now tilt your head up a little." },
  down: { title: "Tilt your head down a little", say: "Now tilt your head down a little." },
};
