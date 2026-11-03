"use client";

// Guided live face capture.
// MediaPipe Face Landmarker runs in the browser (WASM) and tracks the head pose
// in real time, so the instructions react instantly. The backend sets a random
// order of head turns (the "challenge"); a frame is captured automatically when
// each pose is held. The server then re-checks every frame with its own face
// models, matches the face with the ID photo, and an LLM reviews the frames for
// spoofing.
import { useCallback, useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { api, ApiError } from "@/lib/api";
import { poseFromLandmarks, stepProgress, stepReached, STEP_TEXT, type Pose } from "@/lib/headPose";
import type { Challenge, LivenessResult, PoseStep, Submission } from "@/lib/types";

const WASM_PATH = "/mediapipe/wasm";
const LOCAL_MODEL = process.env.NEXT_PUBLIC_FACE_LANDMARKER_URL || "/mediapipe/face_landmarker.task";
const REMOTE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HOLD_MS = 600;
const SLOW_STEP_MS = 20000;

type Phase = "intro" | "loading" | "capturing" | "uploading" | "retry" | "error";

interface Capture {
  step: PoseStep;
  blob: Blob;
  url: string;
}

async function loadLandmarker(): Promise<FaceLandmarker> {
  const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  // Use the self-hosted model when it was downloaded at build time, otherwise Google's copy.
  const localOk = await fetch(LOCAL_MODEL, { method: "HEAD" }).then((r) => r.ok).catch(() => false);
  const model = localOk ? LOCAL_MODEL : REMOTE_MODEL;
  const attempts: Array<[string, "GPU" | "CPU"]> = [[model, "GPU"], [model, "CPU"]];
  let lastError: unknown;
  for (const [modelAssetPath, delegate] of attempts) {
    try {
      return await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath, delegate },
        runningMode: "VIDEO",
        numFaces: 2,
      });
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError ?? new Error("Face tracker failed to load");
}

async function openCamera(): Promise<MediaStream> {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
  } catch (err) {
    // Some webcams/drivers reject the preferred settings; retry with "any camera".
    const name = err instanceof DOMException ? err.name : "";
    if (name === "OverconstrainedError" || name === "NotFoundError") {
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    throw err;
  }
}

function cameraErrorMessage(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError") return "Camera access was blocked. Allow the camera for this site and try again.";
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return "Your browser can't see any camera. Check that the webcam is switched on (many laptops have a " +
      "camera key or privacy switch) and that it works in another app, then try again.";
  if (name === "NotReadableError") return "The camera is being used by another app. Close it and try again.";
  return err instanceof Error ? `${err.name}: ${err.message}` : "Could not start the camera.";
}

export default function FaceCaptureStep({ submission, onAccepted, onEditDocuments }: {
  submission: Submission; onAccepted: () => void; onEditDocuments: () => void;
}) {
  const [phase, setPhase] = useState<Phase>("intro");
  const [steps, setSteps] = useState<PoseStep[]>([]);
  const [stepIndex, setStepIndex] = useState(0);
  const [hint, setHint] = useState("");
  const [progress, setProgress] = useState(0);
  const [good, setGood] = useState(false);
  const [captures, setCaptures] = useState<Capture[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LivenessResult | null>(null);
  const [voice, setVoice] = useState(true);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const loop = useRef({
    challenge: null as Challenge | null,
    index: 0,
    base: null as Pose | null,
    holdStart: null as number | null,
    stepStart: 0,
    lastVideoTime: -1,
    lastUi: 0,
    captures: [] as Capture[],
    busy: false,
  });
  const voiceRef = useRef(voice);
  voiceRef.current = voice;

  const speak = useCallback((text: string) => {
    if (!voiceRef.current || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    window.speechSynthesis.speak(u);
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => () => {
    stopCamera();
    landmarkerRef.current?.close();
    if (typeof window !== "undefined" && "speechSynthesis" in window) window.speechSynthesis.cancel();
    loop.current.captures.forEach((c) => URL.revokeObjectURL(c.url));
  }, [stopCamera]);

  const upload = useCallback(async () => {
    const st = loop.current;
    if (!st.challenge) return;
    setPhase("uploading");
    speak("Thanks. Checking your capture.");
    const form = new FormData();
    form.append("challenge_id", st.challenge.challenge_id);
    st.captures.forEach((c) => form.append(`frame_${c.step}`, c.blob, `${c.step}.jpg`));
    try {
      const res = await api.submitLiveness(submission.id, form);
      setResult(res);
      if (res.accepted) onAccepted();
      else {
        setPhase("retry");
        speak("We could not confirm the movement. Please try again.");
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed. Please try again.");
      setPhase("retry");
    }
  }, [onAccepted, speak, submission.id]);

  const captureFrame = useCallback((step: PoseStep): Promise<Capture> => {
    const video = videoRef.current!;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")!.drawImage(video, 0, 0); // unmirrored, as the server expects
    return new Promise((resolve, reject) =>
      canvas.toBlob((blob) => (blob ? resolve({ step, blob, url: URL.createObjectURL(blob) }) : reject()),
        "image/jpeg", 0.92));
  }, []);

  const tick = useCallback(() => {
    const st = loop.current;
    const video = videoRef.current;
    const lm = landmarkerRef.current;
    const challenge = st.challenge;
    if (!video || !lm || !challenge) return;
    rafRef.current = requestAnimationFrame(() => tickRef.current());
    if (st.busy || video.readyState < 2 || video.currentTime === st.lastVideoTime) return;
    st.lastVideoTime = video.currentTime;

    const now = performance.now();
    const step = challenge.steps[st.index];
    const res = lm.detectForVideo(video, now);
    const faces = res.faceLandmarks ?? [];
    let message = "";
    let reached = false;
    let prog = 0;

    if (faces.length === 0) message = "We can't see your face — look at the camera";
    else if (faces.length > 1) message = "Only one person should be in the frame";
    else {
      const pose = poseFromLandmarks(faces[0], video.videoWidth, video.videoHeight);
      if (pose.faceWidth < 0.17) message = "Move a little closer";
      else if (pose.faceWidth > 0.8) message = "Move back a little";
      else if (step === "center" && (Math.abs(pose.centerX - 0.5) > 0.18 || Math.abs(pose.centerY - 0.45) > 0.2))
        message = "Center your face in the oval";
      else {
        reached = stepReached(step, pose, st.base);
        prog = stepProgress(step, pose, st.base);
        message = reached ? "Hold still…" : STEP_TEXT[step].title;
        if (reached) {
          if (st.holdStart === null) st.holdStart = now;
          else if (now - st.holdStart >= HOLD_MS) {
            st.busy = true;
            st.holdStart = null;
            captureFrame(step).then((cap) => {
              if (step === "center") st.base = pose;
              st.captures = [...st.captures, cap];
              setCaptures(st.captures);
              st.index += 1;
              st.stepStart = performance.now();
              if (st.index >= challenge.steps.length) {
                stopCamera();
                void upload();
              } else {
                setStepIndex(st.index);
                speak(STEP_TEXT[challenge.steps[st.index]].say);
              }
              st.busy = false;
            }).catch(() => { st.busy = false; });
          }
        }
      }
    }
    if (!reached) st.holdStart = null;
    // Only replace the plain instruction; framing hints ("move closer") are more useful.
    if (!reached && now - st.stepStart > SLOW_STEP_MS && message === STEP_TEXT[step].title) {
      message = step === "center"
        ? "Make sure your face is well lit and look straight ahead"
        : `${STEP_TEXT[step].title} — a bit further`;
    }

    if (now - st.lastUi > 100) {
      st.lastUi = now;
      setHint(message);
      setProgress(reached ? 1 : prog);
      setGood(reached);
    }
  }, [captureFrame, speak, stopCamera, upload]);
  const tickRef = useRef(tick);
  tickRef.current = tick;

  // The <video> element exists only in the "capturing" phase: attach the stream once it mounts.
  useEffect(() => {
    if (phase !== "capturing") return;
    const video = videoRef.current;
    const stream = streamRef.current;
    const challenge = loop.current.challenge;
    if (!video || !stream || !challenge) return;
    video.srcObject = stream;
    video.play().catch(() => undefined);
    loop.current.stepStart = performance.now();
    speak(STEP_TEXT[challenge.steps[loop.current.index]].say);
    rafRef.current = requestAnimationFrame(() => tickRef.current());
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [phase, speak]);

  async function start() {
    setError(null);
    setResult(null);
    loop.current.captures.forEach((c) => URL.revokeObjectURL(c.url));
    Object.assign(loop.current, { index: 0, base: null, holdStart: null, captures: [], busy: false, lastVideoTime: -1 });
    setCaptures([]);
    setStepIndex(0);
    setPhase("loading");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser cannot open the camera here. Use a recent Chrome, Safari or Firefox over HTTPS or localhost.");
      setPhase("error");
      return;
    }
    try {
      const [challenge, stream, landmarker] = await Promise.all([
        api.challenge(submission.id),
        openCamera(),
        landmarkerRef.current ? Promise.resolve(landmarkerRef.current) : loadLandmarker(),
      ]);
      landmarkerRef.current = landmarker;
      streamRef.current = stream;
      loop.current.challenge = challenge;
      setSteps(challenge.steps);
      setPhase("capturing");
    } catch (err) {
      stopCamera();
      if (err instanceof ApiError) setError(err.message);
      else if (err instanceof DOMException || (err instanceof Error && /camera|video|Permission/i.test(err.message)))
        setError(cameraErrorMessage(err));
      else setError("The face tracker could not load. Check your connection and try again.");
      setPhase("error");
    }
  }

  const currentStep = steps[stepIndex];

  if (phase === "intro" || phase === "error") {
    return (
      <div className="card">
        <h2>Face check</h2>
        <p className="muted">
          We&apos;ll ask you to look at the camera and slowly turn your head a few ways. Photos are taken
          automatically and compared with the photo on your ID.
        </p>
        <ul className="tips">
          <li>Find good, even light — avoid a bright window behind you.</li>
          <li>Remove sunglasses, hats or masks.</li>
          <li>Hold your phone or sit at eye level with your face inside the oval.</li>
        </ul>
        {error && <div className="alert error" role="alert">{error}</div>}
        <div className="row spread">
          <label className="row small muted" style={{ gap: 6 }}>
            <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice guidance
          </label>
          <div className="row">
            <button className="btn secondary" onClick={onEditDocuments}>Change documents</button>
            <button className="btn" onClick={start}>{phase === "error" ? "Try again" : "Start camera"}</button>
          </div>
        </div>
        <p className="small muted" style={{ marginTop: 16, marginBottom: 0 }}>
          We&apos;re still reading your documents in the background
          {submission.progress ? ` (${submission.progress.toLowerCase()})` : ""}.
        </p>
      </div>
    );
  }

  if (phase === "loading") {
    return (
      <div className="card center-col">
        <div className="spinner" />
        <div>Starting the camera…</div>
        <div className="small muted">Allow camera access when your browser asks.</div>
      </div>
    );
  }

  if (phase === "uploading") {
    return (
      <div className="card center-col">
        <div className="spinner" />
        <div>Checking your face capture…</div>
        <Thumbs steps={steps} captures={captures} current={-1} />
      </div>
    );
  }

  if (phase === "retry") {
    return (
      <div className="card">
        <h2>Let&apos;s try that again</h2>
        {error && <div className="alert error" role="alert">{error}</div>}
        {result && (
          <>
            <ul className="tips">
              {result.issues.map((i) => <li key={i}>{i}</li>)}
            </ul>
            <p className="small muted">
              {result.attempts_left} attempt{result.attempts_left === 1 ? "" : "s"} left.
              Turn your head further and keep your whole face inside the oval.
            </p>
          </>
        )}
        <Thumbs steps={steps} captures={captures} current={-1} />
        <div className="row" style={{ marginTop: 16, justifyContent: "flex-end" }}>
          <button className="btn" onClick={start}>Try again</button>
        </div>
      </div>
    );
  }

  // capturing
  return (
    <div className="card">
      <div className="instruction" aria-live="polite">{currentStep ? STEP_TEXT[currentStep].title : ""}</div>
      <p className="muted" style={{ textAlign: "center" }}>
        Step {Math.min(stepIndex + 1, steps.length)} of {steps.length}
      </p>
      <div className="camera">
        <video ref={videoRef} playsInline muted autoPlay />
        <div className={`oval ${good ? "good" : ""}`} />
        {currentStep === "left" && <div className="arrow left" aria-hidden>←</div>}
        {currentStep === "right" && <div className="arrow right" aria-hidden>→</div>}
        {currentStep === "up" && <div className="arrow up" aria-hidden>↑</div>}
        <div className="overlay-msg" aria-live="polite">{hint}</div>
      </div>
      <div className="meter" aria-hidden><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>
      <Thumbs steps={steps} captures={captures} current={stepIndex} />
      <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
        <button className="btn ghost" onClick={() => { stopCamera(); setPhase("intro"); }}>Cancel</button>
      </div>
    </div>
  );
}

function Thumbs({ steps, captures, current }: { steps: PoseStep[]; captures: Capture[]; current: number }) {
  return (
    <div className="thumbs">
      {steps.map((s, i) => {
        const cap = captures.find((c) => c.step === s);
        return (
          <div key={s} className={`thumb ${cap ? "done" : ""} ${i === current ? "current" : ""}`}>
            {cap ? <img src={cap.url} alt={`${s} capture`} /> : <div className="ph" />}
            {s}
          </div>
        );
      })}
    </div>
  );
}
