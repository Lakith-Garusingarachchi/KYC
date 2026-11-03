// Thin fetch wrapper around the backend (reached through the /api proxy).
import type { Challenge, LivenessResult, Submission, User } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function messageFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length) {
      const first = detail[0] as { msg?: string; loc?: unknown[] };
      const field = Array.isArray(first.loc) ? String(first.loc[first.loc.length - 1]) : "";
      return [field, first.msg].filter(Boolean).join(": ");
    }
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, { credentials: "same-origin", cache: "no-store", ...init });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) throw new ApiError(res.status, messageFrom(body, `Request failed (${res.status})`));
  return body as T;
}

export const api = {
  providers: () => request<{ password: boolean; google: boolean }>("/auth/providers"),
  me: () => request<User>("/auth/me"),
  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username: email, password }),
    }),
  register: (email: string, password: string, full_name: string) =>
    request<{ access_token: string }>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name }),
    }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),

  startSubmission: (fresh = false) =>
    request<Submission>(`/kyc/submissions${fresh ? "?new=true" : ""}`, { method: "POST" }),
  getSubmission: (id: string) => request<Submission>(`/kyc/submissions/${id}`),
  latestSubmission: () => request<Submission>("/kyc/submissions/current"),
  uploadDocuments: (id: string, form: FormData) =>
    request<Submission>(`/kyc/submissions/${id}/documents`, { method: "POST", body: form }),
  challenge: (id: string) => request<Challenge>(`/kyc/submissions/${id}/liveness/challenge`, { method: "POST" }),
  submitLiveness: (id: string, form: FormData) =>
    request<LivenessResult>(`/kyc/submissions/${id}/liveness`, { method: "POST", body: form }),
};
