import ChatWorkspace from './chat-workspace';
import AuthGate from './auth-gate';

export default function Home() {
  return <AuthGate><ChatWorkspace /></AuthGate>;
}
