"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import type { Submission, User } from "@/lib/types";
import DocumentsStep from "@/components/DocumentsStep";
import FaceCaptureStep from "@/components/FaceCaptureStep";
import ResultStep from "@/components/ResultStep";

const STEPS = ["Documents", "Face check", "Result"];

function stepFor(sub: Submission, editingDocs: boolean): number {
  if (sub.status === "draft" || editingDocs) return 0;
  if (sub.status === "documents_uploaded") return 1;
  return 2;
}

export default function KycPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [sub, setSub] = useState<Submission | null>(null);
  const [editingDocs, setEditingDocs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setUser(await api.me());
        // Show the latest verification (including a finished result); start one if there is none.
        try {
          setSub(await api.latestSubmission());
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) setSub(await api.startSubmission());
          else throw err;
        }
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) router.replace("/login");
        else setError(err instanceof Error ? err.message : "Could not load your verification");
      }
    })();
  }, [router]);

  const refresh = useCallback(async () => {
    if (!sub) return;
    setSub(await api.getSubmission(sub.id));
  }, [sub]);

  async function logout() {
    await api.logout().catch(() => undefined);
    router.replace("/login");
  }

  async function startOver() {
    setEditingDocs(false);
    setSub(await api.startSubmission(true));
  }

  const step = sub ? stepFor(sub, editingDocs) : -1;

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden>ID</div>
          <span>Identity Verification</span>
        </div>
        {user && (
          <div className="topbar-user">
            <span>{user.email}</span>
            <button className="btn ghost" onClick={logout}>Log out</button>
          </div>
        )}
      </header>

      <main className="shell">
        {sub && (
          <ol className="stepper" aria-label="Progress">
            {STEPS.map((label, i) => (
              <li key={label} className={i < step ? "done" : i === step ? "active" : ""}
                  aria-current={i === step ? "step" : undefined}>
                <span className="dot">{i < step ? "✓" : i + 1}</span>
                <span className="label">{label}</span>
              </li>
            ))}
          </ol>
        )}

        {error && <div className="alert error">{error}</div>}
        {!sub && !error && (
          <div className="center-col"><div className="spinner" /><span className="muted">Loading…</span></div>
        )}

        {sub && step === 0 && (
          <DocumentsStep
            submission={sub}
            onUploaded={(s) => { setEditingDocs(false); setSub(s); }}
          />
        )}
        {sub && step === 1 && (
          <FaceCaptureStep
            submission={sub}
            onAccepted={refresh}
            onEditDocuments={() => setEditingDocs(true)}
          />
        )}
        {sub && step === 2 && <ResultStep submission={sub} onChange={setSub} onStartOver={startOver} />}
      </main>
    </>
  );
}
