"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, ApiError } from "@/lib/api";

const OAUTH_ERRORS: Record<string, string> = {
  oauth_state: "Google sign-in expired. Please try again.",
  oauth_exchange: "Google sign-in failed. Please try again.",
  oauth_email: "Your Google account email is not verified.",
};

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [mode, setMode] = useState<"signin" | "register">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(OAUTH_ERRORS[params.get("error") ?? ""] ?? null);
  const [busy, setBusy] = useState(false);
  const [google, setGoogle] = useState(false);

  useEffect(() => {
    api.me().then(() => router.replace("/kyc")).catch(() => undefined);
    api.providers().then((p) => setGoogle(p.google)).catch(() => undefined);
  }, [router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "signin") await api.login(email, password);
      else await api.register(email, password, fullName);
      router.replace("/kyc");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      setBusy(false);
    }
  }

  return (
    <main className="auth-wrap">
      <div className="card auth-card">
        <div className="brand" style={{ marginBottom: 12 }}>
          <div className="brand-mark" aria-hidden>ID</div>
          <span>Identity Verification</span>
        </div>
        <h1>{mode === "signin" ? "Sign in" : "Create your account"}</h1>
        <p className="muted">
          Verify your identity with your NIC or passport, a recent utility bill and a quick face check.
        </p>

        <div className="tabs" role="tablist">
          <button role="tab" type="button" aria-selected={mode === "signin"} onClick={() => setMode("signin")}>
            Sign in
          </button>
          <button role="tab" type="button" aria-selected={mode === "register"} onClick={() => setMode("register")}>
            Create account
          </button>
        </div>

        {error && <div className="alert error" role="alert">{error}</div>}

        <form onSubmit={submit}>
          {mode === "register" && (
            <label className="field">
              <span>Full name</span>
              <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)}
                     autoComplete="name" placeholder="As shown on your ID" />
            </label>
          )}
          <label className="field">
            <span>Email</span>
            <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                   autoComplete="email" placeholder="you@example.com" />
          </label>
          <label className="field">
            <span>Password</span>
            <input type="password" required minLength={8} value={password}
                   onChange={(e) => setPassword(e.target.value)}
                   autoComplete={mode === "signin" ? "current-password" : "new-password"} />
            {mode === "register" && <small>At least 8 characters.</small>}
          </label>
          <button className="btn block" type="submit" disabled={busy}>
            {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>

        {google && (
          <>
            <div className="divider">or</div>
            <a className="btn secondary block" href="/api/auth/google/login">Continue with Google</a>
          </>
        )}
      </div>
    </main>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}
