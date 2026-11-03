"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Submission } from "@/lib/types";

const MAX_MB = 10;
const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp";

type Slot = "id_front" | "id_back" | "utility_bill";

function FileDrop({ title, hint, accept, file, onChange, required }: {
  title: string; hint: string; accept: string; file: File | null; required?: boolean;
  onChange: (f: File | null) => void;
}) {
  const [preview, setPreview] = useState<string | null>(null);
  useEffect(() => {
    if (!file || !file.type.startsWith("image/")) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <label className={`drop ${file ? "filled" : ""}`}>
      <input type="file" accept={accept} required={required && !file}
             onChange={(e) => onChange(e.target.files?.[0] ?? null)} aria-label={title} />
      {preview ? <img src={preview} alt={`${title} preview`} /> : <div style={{ fontSize: 26, fontWeight: 700 }} aria-hidden>{file ? "PDF" : "+"}</div>}
      <div className="title">{title}{required ? "" : " (optional)"}</div>
      <div className="hint">{file ? file.name : hint}</div>
    </label>
  );
}

export default function DocumentsStep({ submission, onUploaded }: {
  submission: Submission; onUploaded: (s: Submission) => void;
}) {
  const [idType, setIdType] = useState<"nic" | "passport">(submission.id_type ?? "nic");
  const [files, setFiles] = useState<Record<Slot, File | null>>({ id_front: null, id_back: null, utility_bill: null });
  const [address, setAddress] = useState(submission.address_input ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const setFile = (slot: Slot) => (f: File | null) => {
    if (f && f.size > MAX_MB * 1024 * 1024) {
      setError(`${f.name} is larger than ${MAX_MB} MB`);
      return;
    }
    setError(null);
    setFiles((prev) => ({ ...prev, [slot]: f }));
  };

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!files.id_front || !files.utility_bill) {
      setError("Add your ID and a utility bill.");
      return;
    }
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("id_type", idType);
    form.append("address", address.trim());
    form.append("id_front", files.id_front);
    if (files.id_back) form.append("id_back", files.id_back);
    form.append("utility_bill", files.utility_bill);
    try {
      onUploaded(await api.uploadDocuments(submission.id, form));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed. Please try again.");
      setBusy(false);
    }
  }

  return (
    <form className="card" onSubmit={submit}>
      <h2>Your documents</h2>
      <p className="muted">
        Take clear, well-lit photos of the whole document, with all four corners visible and no glare.
      </p>
      {error && <div className="alert error" role="alert">{error}</div>}

      <div className="field"><span>Identity document</span></div>
      <div className="choices" role="radiogroup" aria-label="Identity document type">
        {(["nic", "passport"] as const).map((t) => (
          <label key={t} className={`choice ${idType === t ? "selected" : ""}`}>
            <input type="radio" name="id_type" value={t} checked={idType === t} onChange={() => setIdType(t)} />
            <div>
              <div style={{ fontWeight: 600 }}>{t === "nic" ? "National ID card (NIC)" : "Passport"}</div>
              <div className="small muted">{t === "nic" ? "Front and back" : "Photo page"}</div>
            </div>
          </label>
        ))}
      </div>

      <div className="drops">
        <FileDrop title={idType === "nic" ? "NIC front" : "Passport photo page"} hint="JPG, PNG or WEBP"
                  accept={IMAGE_ACCEPT} file={files.id_front} onChange={setFile("id_front")} required />
        {idType === "nic" && (
          <FileDrop title="NIC back" hint="JPG, PNG or WEBP" accept={IMAGE_ACCEPT}
                    file={files.id_back} onChange={setFile("id_back")} />
        )}
        <FileDrop title="Utility bill" hint="Electricity, water, phone or internet — last 3 months. Image or PDF"
                  accept={`${IMAGE_ACCEPT},application/pdf`} file={files.utility_bill}
                  onChange={setFile("utility_bill")} required />
      </div>

      <label className="field">
        <span>Your full residential address</span>
        <textarea required minLength={5} maxLength={500} value={address} onChange={(e) => setAddress(e.target.value)}
                  placeholder="House number, street, city, postal code" autoComplete="street-address" />
        <small>Type it as it appears on your utility bill.</small>
      </label>

      <div className="row spread">
        <span className="small muted">Files are encrypted when stored.</span>
        <button className="btn" type="submit" disabled={busy}>{busy ? "Uploading…" : "Continue to face check"}</button>
      </div>
    </form>
  );
}
