'use client';

import { useEffect, useState } from 'react';
import { googleFirebaseIdToken } from './firebase-auth';

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://127.0.0.1:8000';

type Identity = { uid: string; email?: string; is_admin: boolean };

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [checking, setChecking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    void fetch(`${API_URL}/auth/me`, { credentials: 'include' })
      .then(async (response) => {
        if (response.ok) setIdentity(await response.json() as Identity);
      })
      .catch(() => setError('The RAG server is not reachable.'))
      .finally(() => setChecking(false));
  }, []);

  async function signIn() {
    setBusy(true);
    setError('');
    try {
      const idToken = await googleFirebaseIdToken();
      const response = await fetch(`${API_URL}/auth/session`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id_token: idToken }),
      });
      const payload = (await response.json().catch(() => ({}))) as
        Partial<Identity> & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || 'Google sign-in failed.');
      setIdentity(payload as Identity);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Google sign-in failed.');
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    await fetch(`${API_URL}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    }).catch(() => undefined);
    setIdentity(null);
  }

  if (checking) return <main className="auth-screen"><p>Opening your study workspace…</p></main>;
  if (!identity) return (
    <main className="auth-screen">
      <section className="auth-card">
        <span className="brand-mark">स</span>
        <p className="eyebrow">SARTHI AI</p>
        <h1>Your private study workspace</h1>
        <p>Sign in to access only your own documents, conversations, and generated visuals.</p>
        <button type="button" onClick={() => void signIn()} disabled={busy}>
          {busy ? 'Signing in…' : 'Continue with Google'}
        </button>
        {error && <p className="auth-error" role="alert">{error}</p>}
      </section>
    </main>
  );

  return <><button className="auth-signout" type="button" onClick={() => void signOut()} title={identity.email}>Sign out</button>{children}</>;
}
