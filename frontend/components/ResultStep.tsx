"use client";

import { useEffect } from "react";
import { api } from "@/lib/api";
import type { Check, Submission } from "@/lib/types";

const POLL_MS = 2000;

const DECISION_TEXT = {
  approved: { icon: "✓", title: "Passed", body: "The verification score is at or above the pass mark." },
  manual_review: {
    icon: "!",
    title: "Manual review required",
    body: "A person needs to look at this verification before it can pass.",
  },
  // Results stored before the score was introduced.
  rejected: { icon: "!", title: "Manual review required", body: "A person needs to look at this verification." },
} as const;

function ScoreMeter({ score, passScore }: { score: number; passScore: number }) {
  const ok = score >= passScore;
  return (
    <div className="score">
      <div className="score-head">
        <span>Verification score</span>
        <strong className={ok ? "pass-text" : "warn-text"}>{score}%</strong>
      </div>
      <div className="score-bar" role="img" aria-label={`Score ${score}% of 100, pass mark ${passScore}%`}>
        <div className={`score-fill ${ok ? "ok" : "low"}`} style={{ width: `${score}%` }} />
        <div className="score-mark" style={{ left: `${passScore}%` }} title={`Pass mark ${passScore}%`} />
      </div>
      <div className="small muted">Pass mark {passScore}% · review counts as half a pass</div>
    </div>
  );
}

const CATEGORY_TITLES: Record<string, string> = {
  document: "Identity document",
  address: "Address & utility bill",
  liveness: "Liveness",
  face: "Face match",
};

function Facts({ title, rows }: { title: string; rows: Array<[string, string | null | undefined]> }) {
  const shown = rows.filter(([, v]) => v);
  if (!shown.length) return null;
  return (
    <div>
      <h3>{title}</h3>
      <dl>
        {shown.map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function ResultStep({ submission, onChange, onStartOver }: {
  submission: Submission; onChange: (s: Submission) => void; onStartOver: () => void;
}) {
  const pending = submission.status === "processing" || submission.status === "documents_uploaded";

  useEffect(() => {
    if (!pending) return;
    const timer = setInterval(async () => {
      try {
        onChange(await api.getSubmission(submission.id));
      } catch {
        /* keep polling */
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [pending, submission.id, onChange]);

  if (pending) {
    return (
      <div className="card center-col" aria-live="polite">
        <div className="spinner" />
        <h2 style={{ margin: 0 }}>Verifying…</h2>
        <div className="muted">{submission.progress || "Running checks"}</div>
        <div className="small muted">This usually takes under a minute.</div>
      </div>
    );
  }

  if (submission.status === "failed") {
    return (
      <div className="card">
        <h2>Something went wrong</h2>
        <div className="alert error">{submission.error || "The verification could not be completed."}</div>
        <button className="btn" onClick={onStartOver}>Start again</button>
      </div>
    );
  }

  const decision = submission.decision ?? "manual_review";
  const text = DECISION_TEXT[decision];
  const groups = submission.checks.reduce<Record<string, Check[]>>((acc, c) => {
    (acc[c.category] ||= []).push(c);
    return acc;
  }, {});
  const issues = submission.checks.filter((c) => c.status !== "pass");
  const idd = submission.extracted.identity ?? {};
  const bill = submission.extracted.bill ?? {};
  const match = submission.face.face_match;

  return (
    <>
      <div className="card report-head">
        <div className={`decision ${decision === "rejected" ? "manual_review" : decision}`}>
          <div className="icon" aria-hidden>{text.icon}</div>
          <div>
            <h2 style={{ marginBottom: 4 }}>{text.title}</h2>
            <div className="muted">{text.body}</div>
          </div>
        </div>
        {submission.score != null && submission.pass_score != null && (
          <ScoreMeter score={submission.score} passScore={submission.pass_score} />
        )}
        {decision !== "approved" && (submission.blockers.length > 0 ||
          (submission.score != null && submission.pass_score != null && submission.score < submission.pass_score)) && (
          <div className="alert warn" style={{ marginTop: 16, marginBottom: 0 }}>
            <strong>Why it needs review:</strong>
            <ul className="tips" style={{ margin: "6px 0 0", color: "inherit" }}>
              {submission.score != null && submission.pass_score != null && submission.score < submission.pass_score && (
                <li>Score {submission.score}% is below the pass mark of {submission.pass_score}%</li>
              )}
              {submission.blockers.map((b) => <li key={b}>Required check not passed: {b}</li>)}
            </ul>
          </div>
        )}
        {decision === "approved" && issues.length > 0 && (
          <div className="alert info" style={{ marginTop: 16, marginBottom: 0 }}>
            <strong>Points to note:</strong>
            <ul className="tips" style={{ margin: "6px 0 0", color: "inherit" }}>
              {issues.map((c) => <li key={c.id}>{c.label}{c.status === "fail" ? " (failed)" : " (review)"}</li>)}
            </ul>
          </div>
        )}
        <div className="small muted report-meta">
          Report {submission.id.slice(0, 8).toUpperCase()} · {new Date(submission.updated_at).toLocaleString()}
        </div>
      </div>

      <div className="card">
        <h2>Checks</h2>
        {Object.entries(groups).map(([cat, checks]) => (
          <div key={cat} style={{ marginTop: 12 }}>
            <h3 className="muted">{CATEGORY_TITLES[cat] ?? cat}</h3>
            <ul className="checks">
              {checks.map((c) => (
                <li key={c.id}>
                  <span className={`badge ${c.status}`}>{c.status === "warn" ? "review" : c.status}</span>
                  <div>
                    <div className="check-label">{c.label}</div>
                    {c.detail && <div className="check-detail">{c.detail}</div>}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="card facts">
        <Facts title="From your ID" rows={[
          ["Name", idd.full_name], ["Number", idd.id_number], ["Date of birth", idd.date_of_birth],
          ["Gender", idd.gender === "unknown" ? null : idd.gender], ["Address", idd.address],
          ["Expires", idd.expiry_date],
        ]} />
        <Facts title="From your bill" rows={[
          ["Issuer", bill.issuer], ["Account holder", bill.account_holder_name],
          ["Address", bill.service_address], ["Bill date", bill.bill_date],
        ]} />
        <Facts title="Face" rows={[
          ["Liveness", submission.face.liveness_pass ? "Passed" : "Not passed"],
          ["Match score", match?.score != null ? match.score.toFixed(2) : null],
          ["You typed", submission.address_input],
        ]} />
      </div>

      <div className="row no-print" style={{ justifyContent: "flex-end", marginTop: 16 }}>
        <button className="btn secondary" onClick={() => window.print()}>Download report (PDF)</button>
        <button className="btn secondary" onClick={onStartOver}>Start a new verification</button>
      </div>
    </>
  );
}
