'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { googleFirebaseIdToken } from './firebase-auth';

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000';

export type Identity = { uid: string; email?: string; is_admin: boolean };

type AuthState = {
  identity: Identity | null;
  checking: boolean;
  busy: boolean;
  error: string;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthState>({
  identity: null,
  checking: true,
  busy: false,
  error: '',
  signIn: async () => undefined,
  signOut: async () => undefined,
});

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

  const signIn = useCallback(async () => {
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
  }, []);

  const signOut = useCallback(async () => {
    await fetch(`${API_URL}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    }).catch(() => undefined);
    setIdentity(null);
  }, []);

  const authState = useMemo(() => ({ identity, checking, busy, error, signIn, signOut }), [identity, checking, busy, error, signIn, signOut]);

  return (
    <AuthContext.Provider value={authState}>
      {children}
    </AuthContext.Provider>
  );
}

export function AuthSession() {
  const { identity, checking, busy, error, signIn, signOut } = useAuth();

  return (
    <div className={`auth-session${error ? ' has-error' : ''}`} aria-live="polite">
      {identity ? (
        <>
          <span className="auth-identity">
            {identity.is_admin && <span className="auth-role">Administrator</span>}
            <span>Signed in as:</span>
            <span className="auth-email" title={identity.email}>
              {identity.email || 'your account'}
            </span>
          </span>
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
  );
}
