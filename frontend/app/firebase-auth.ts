type FirebaseModule = Record<string, (...args: unknown[]) => unknown>;

const SDK_VERSION = process.env.NEXT_PUBLIC_FIREBASE_SDK_VERSION || '12.2.1';

async function loadModule(name: 'firebase-app' | 'firebase-auth') {
  const url = `https://www.gstatic.com/firebasejs/${SDK_VERSION}/${name}.js`;
  return import(/* @vite-ignore */ url) as Promise<FirebaseModule>;
}

export async function googleFirebaseIdToken(): Promise<string> {
  const [appModule, authModule] = await Promise.all([
    loadModule('firebase-app'),
    loadModule('firebase-auth'),
  ]);
  const config = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
  };
  if (!config.apiKey || !config.authDomain || !config.projectId || !config.appId) {
    throw new Error('Firebase sign-in is not configured for this deployment.');
  }
  const apps = appModule.getApps() as unknown[];
  const app = apps.length ? apps[0] : appModule.initializeApp(config);
  const firebaseAuth = authModule.getAuth(app);
  const provider = new (authModule.GoogleAuthProvider as unknown as new () => unknown)();
  const credential = await authModule.signInWithPopup(firebaseAuth, provider) as {
    user: { getIdToken: () => Promise<string> };
  };
  const idToken = await credential.user.getIdToken();
  await authModule.signOut(firebaseAuth);
  return idToken;
}
