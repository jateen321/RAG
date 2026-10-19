'use client';

import { createContext, useContext } from 'react';

export type Identity = { uid: string; email?: string; is_admin: boolean };

type AuthState = {
  identity: Identity;
  checking: boolean;
  busy: boolean;
  error: string;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
};

const SINGLE_USER: Identity = {
  uid: 'single-user',
  email: 'Single user',
  is_admin: true,
};

const AUTH_STATE: AuthState = {
  identity: SINGLE_USER,
  checking: false,
  busy: false,
  error: '',
  signIn: async () => undefined,
  signOut: async () => undefined,
};

const AuthContext = createContext<AuthState>(AUTH_STATE);

export function useAuth() {
  return useContext(AuthContext);
}

export default function AuthGate({ children }: { children: React.ReactNode }) {
  return <AuthContext.Provider value={AUTH_STATE}>{children}</AuthContext.Provider>;
}

export function AuthSession() {
  return (
    <div className="auth-session" aria-live="polite">
      <span className="auth-identity">Single-user mode</span>
    </div>
  );
}
