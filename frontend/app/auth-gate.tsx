'use client';

import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { googleFirebaseIdToken } from './firebase-auth';

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000';

export type Identity = { uid: string; email?: string; is_admin: boolean };

type AuthState = {
  identity: Identity | null;
  checking: boolean;
};

const AuthContext = createContext<AuthState>({ identity: null, checking: true });

export function useAuth() {
  return useContext(AuthContext);
}

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

  const authState = useMemo(() => ({ identity, checking }), [identity, checking]);

  return (
    <AuthContext.Provider value={authState}>
      <div className="auth-session" aria-live="polite">
        {identity ? (
          <>
            <span title={identity.email}>{identity.is_admin ? 'Administrator' : 'Signed in'}</span>
            <button type="button" onClick={() => void signOut()}>Sign out</button>
          </>
        ) : (
          <>
            <span>{checking ? 'Checking session…' : 'Guest access'}</span>
            <button type="button" onClick={() => void signIn()} disabled={checking || busy}>
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </>
        )}
        {error && <span className="auth-error" role="alert">{error}</span>}
      </div>
      {children}
    </AuthContext.Provider>
  );
}
