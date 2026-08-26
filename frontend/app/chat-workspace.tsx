'use client';

import { ChangeEvent, FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://127.0.0.1:8000';

type DocumentInfo = {
  source: string;
  chunks: number;
  pages?: number;
  source_type?: string;
};

type HealthResponse = {
  status: string;
  total_chunks: number;
  documents: DocumentInfo[];
};

type Source = {
  page?: number;
  source: string;
  distance: number;
  preview: string;
  source_type?: string;
  timestamp?: string;
  source_url?: string;
  video_title?: string;
};

type AskResponse = {
  answer: string;
  sources: Source[];
  timings?: { total_s?: number };
};

type Conversation = {
  id: number;
  question: string;
  answer?: string;
  sources?: Source[];
  totalSeconds?: number;
  pending?: boolean;
  error?: string;
};

type Notice = { tone: 'success' | 'error'; text: string } | null;

const suggestions = [
  'Summarize the main ideas in the indexed books',
  "What does the CIL report say about India's coal production?",
  'भाग्य और कर्म के संबंध को समझाइए',
];

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error('The RAG server is not reachable. Start the FastAPI server and try again.');
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return payload as T;
}

function BookIcon({ youtube = false }: { youtube?: boolean }) {
  if (youtube) return <span className="video-icon" aria-hidden="true">▶</span>;
  return <span className="book-icon" aria-hidden="true"><span /><span /></span>;
}

function locationLabel(source: Source) {
  if (source.source_type === 'youtube') return source.timestamp ? `Timestamp ${source.timestamp}` : 'Video transcript';
  if (source.source_type === 'text' || source.source_type === 'markdown') return source.page ? `Document section ${source.page}` : 'Document passage';
  return source.page ? `Page ${source.page}` : 'PDF passage';
}

function LoadingAnswer() {
  return (
    <div className="thinking" role="status">
      <span /><span /><span />
      <p>Searching your library and composing a grounded answer…</p>
    </div>
  );
}

export default function ChatWorkspace() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState('');
  const [question, setQuestion] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeSources, setActiveSources] = useState<Source[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileLibraryOpen, setMobileLibraryOpen] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [youtubeOpen, setYoutubeOpen] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [busyAction, setBusyAction] = useState<'upload' | 'youtube' | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const threadEnd = useRef<HTMLDivElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await requestJson<HealthResponse>('/health');
      setHealth(data);
      setHealthError('');
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : 'Could not load the library.');
    }
  }, []);

  useEffect(() => { void refreshHealth(); }, [refreshHealth]);
  useEffect(() => { threadEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [conversations]);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function ask(nextQuestion = question) {
    const cleanQuestion = nextQuestion.trim();
    if (!cleanQuestion || conversations.some((message) => message.pending)) return;

    const id = Date.now();
    setQuestion('');
    setConversations((current) => [...current, { id, question: cleanQuestion, pending: true }]);
    try {
      const data = await requestJson<AskResponse>('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: cleanQuestion }),
      });
      setConversations((current) => current.map((message) => message.id === id ? {
        ...message,
        pending: false,
        answer: data.answer,
        sources: data.sources,
        totalSeconds: data.timings?.total_s,
      } : message));
      setActiveSources(data.sources || []);
    } catch (error) {
      setConversations((current) => current.map((message) => message.id === id ? {
        ...message,
        pending: false,
        error: error instanceof Error ? error.message : 'The question could not be answered.',
      } : message));
    }
  }

  function submitQuestion(event: FormEvent) {
    event.preventDefault();
    void ask();
  }

  function handleQuestionKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void ask();
    }
  }

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] || null;
    if (!file) return;
    if (!['.pdf', '.txt', '.md'].some((extension) => file.name.toLowerCase().endsWith(extension))) {
      setNotice({ tone: 'error', text: 'Choose a PDF, TXT, or Markdown file.' });
      event.target.value = '';
      return;
    }
    if (file.size > 500 * 1024 * 1024) {
      setNotice({ tone: 'error', text: 'Choose a document no larger than 500 MB.' });
      event.target.value = '';
      return;
    }
    setPendingFile(file);
  }

  async function uploadDocument() {
    if (!pendingFile) return;
    setBusyAction('upload');
    const body = new FormData();
    body.append('file', pendingFile);
    try {
      const result = await requestJson<{ source: string; pages_with_text: number; chunks_indexed: number }>('/upload', {
        method: 'POST',
        body,
      });
      setNotice({ tone: 'success', text: `${result.source} indexed with ${result.chunks_indexed} passages.` });
      setPendingFile(null);
      if (fileInput.current) fileInput.current.value = '';
      await refreshHealth();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'The document could not be indexed.' });
    } finally {
      setBusyAction(null);
    }
  }

  async function indexYoutube(event: FormEvent) {
    event.preventDefault();
    if (!youtubeUrl.trim()) return;
    setBusyAction('youtube');
    try {
      const result = await requestJson<{ videos_indexed?: number; chunks_indexed?: number }>('/index/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: youtubeUrl.trim() }),
      });
      setNotice({ tone: 'success', text: `Indexed ${result.videos_indexed ?? 1} video${result.videos_indexed === 1 ? '' : 's'} and ${result.chunks_indexed ?? 0} passages.` });
      setYoutubeUrl('');
      setYoutubeOpen(false);
      await refreshHealth();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'The YouTube source could not be indexed.' });
    } finally {
      setBusyAction(null);
    }
  }

  const hasConversation = conversations.length > 0;
  const isAsking = conversations.some((message) => message.pending);

  function toggleLibrary() {
    if (window.matchMedia('(max-width: 720px)').matches) {
      setMobileLibraryOpen((open) => !open);
      return;
    }
    setSidebarCollapsed((collapsed) => !collapsed);
  }

  return (
    <main className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <aside id="library-sidebar" className={`library-panel ${mobileLibraryOpen ? 'mobile-open' : ''}`}>
        <div className="mobile-panel-head">
          <div className="brand">
            <span className="brand-mark">प</span>
            <div><strong>Pustak AI</strong><span>Grounded study companion</span></div>
          </div>
          <button className="close-panel" type="button" onClick={() => setMobileLibraryOpen(false)} aria-label="Close library">×</button>
        </div>

        <button className="new-chat" type="button" onClick={() => { setConversations([]); setActiveSources([]); setMobileLibraryOpen(false); }}>
          <span aria-hidden="true">＋</span> New conversation
        </button>

        <div className="library-heading">
          <span>Your library</span>
          <span className="library-count">{health?.documents.length ?? '—'}</span>
        </div>

        <div className="document-list">
          {!health && !healthError && <div className="library-loading"><span /><span /><span /></div>}
          {healthError && <button className="inline-error" type="button" onClick={() => void refreshHealth()}>{healthError} <strong>Retry</strong></button>}
          {health?.documents.map((document) => (
            <div className="document-card" key={document.source} title={document.source}>
              <BookIcon youtube={document.source_type === 'youtube'} />
              <span>
                <strong>{document.source}</strong>
                <small>{document.pages ? `${document.pages} pages · ` : ''}{document.chunks} passages</small>
              </span>
            </div>
          ))}
          {health && health.documents.length === 0 && <p className="empty-library">Your library is empty. Add a document or YouTube source to begin.</p>}
        </div>

        <div className="library-actions">
          <button type="button" onClick={() => fileInput.current?.click()}><span aria-hidden="true">↑</span> Add a document</button>
          <button type="button" onClick={() => setYoutubeOpen(true)}><span aria-hidden="true">▶</span> Add YouTube</button>
          <input ref={fileInput} className="visually-hidden" type="file" accept="application/pdf,text/plain,text/markdown,.pdf,.txt,.md" onChange={chooseFile} />
        </div>

        <div className={`index-status ${healthError ? 'offline' : ''}`}>
          <span className="status-dot" />
          <span>
            <strong>{healthError ? 'Server unavailable' : health ? 'Knowledge base ready' : 'Connecting…'}</strong>
            <small>{health ? `${health.total_chunks} passages indexed` : 'Checking your library'}</small>
          </span>
        </div>
      </aside>

      {mobileLibraryOpen && <button className="mobile-scrim" type="button" aria-label="Close library" onClick={() => setMobileLibraryOpen(false)} />}

      <section className="conversation-panel">
        <header className="conversation-header">
          <div className="conversation-title">
            <button className="sidebar-toggle" type="button" onClick={toggleLibrary} aria-label="Toggle library sidebar" aria-controls="library-sidebar">
              <span /><span /><span />
            </button>
            <div><span className="eyebrow">STUDY WORKSPACE</span><h1>Ask your books</h1></div>
          </div>
        </header>

        {!hasConversation ? (
          <div className="welcome-state">
            <div className="welcome-seal">अ</div>
            <p className="eyebrow">YOUR INDEXED KNOWLEDGE</p>
            <h2>Read less. Understand more.</h2>
            <p className="welcome-copy">Ask in English or Hindi. Every answer is grounded in your documents and linked back to its source.</p>
            <div className="suggestion-list">
              {suggestions.map((suggestion) => (
                <button key={suggestion} type="button" onClick={() => void ask(suggestion)} disabled={isAsking}>
                  <span>{suggestion}</span><span aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="conversation-thread" aria-live="polite">
            {conversations.map((message) => (
              <article className="exchange" key={message.id}>
                <div className="question-bubble"><span>You</span><p>{message.question}</p></div>
                <div className="answer-block">
                  <div className="assistant-badge">प</div>
                  <div className="answer-content">
                    <div className="answer-heading"><strong>Pustak AI</strong>{message.totalSeconds != null && <small>{message.totalSeconds.toFixed(1)}s · {message.sources?.length ?? 0} sources</small>}</div>
                    {message.pending && <LoadingAnswer />}
                    {message.error && <div className="answer-error"><strong>I couldn’t answer that.</strong><p>{message.error}</p><button type="button" onClick={() => void ask(message.question)}>Try again</button></div>}
                    {message.answer && <div className="answer-copy">{message.answer}</div>}
                    {!!message.sources?.length && (
                      <>
                        <button className="show-sources" type="button" onClick={() => setActiveSources(message.sources || [])}>
                          View {message.sources.length} supporting sources <span aria-hidden="true">→</span>
                        </button>
                        <div className="inline-sources">
                          {message.sources.map((source, index) => (
                            <div key={`${source.source}-${index}`}><strong>{index + 1}. {locationLabel(source)}</strong><span>{source.source}</span></div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </article>
            ))}
            <div ref={threadEnd} />
          </div>
        )}

        <form className="composer-wrap" onSubmit={submitQuestion}>
          <div className="composer">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleQuestionKey} aria-label="Ask a question" placeholder="Ask a question about your books…" rows={1} maxLength={2000} />
            <div className="composer-footer">
              <span>Enter to send · Shift + Enter for a new line</span>
              <button type="submit" aria-label="Send question" disabled={!question.trim() || isAsking}>{isAsking ? '…' : '↑'}</button>
            </div>
          </div>
          <p>Answers are generated from retrieved passages. Always verify important details.</p>
        </form>
      </section>

      <aside className="context-panel">
        {activeSources.length ? (
          <>
            <p className="eyebrow">SUPPORTING EVIDENCE</p>
            <h2>Sources for this answer.</h2>
            <div className="source-list">
              {activeSources.map((source, index) => {
                const content = (
                  <>
                    <div className="source-card-head"><span>{String(index + 1).padStart(2, '0')}</span><strong>{locationLabel(source)}</strong></div>
                    <h3>{source.video_title || source.source}</h3>
                    <p>{source.preview}</p>
                    <small>{source.source_type === 'youtube' ? 'YouTube transcript' : source.source}</small>
                  </>
                );
                return source.source_url ? <a className="source-card" href={source.source_url} target="_blank" rel="noreferrer" key={`${source.source}-${index}`}>{content}</a> : <article className="source-card" key={`${source.source}-${index}`}>{content}</article>;
              })}
            </div>
          </>
        ) : (
          <>
            <p className="eyebrow">HOW IT WORKS</p>
            <h2>Answers you can trace.</h2>
            <div className="process-list">
              <div><span>01</span><p><strong>Ask naturally</strong><small>Use Hindi or English—just like talking to a tutor.</small></p></div>
              <div><span>02</span><p><strong>Find the evidence</strong><small>The five closest passages are retrieved from your library.</small></p></div>
              <div><span>03</span><p><strong>Check every source</strong><small>Answers include page numbers, timestamps, and text previews.</small></p></div>
            </div>
            <div className="source-note"><span className="quote-mark">“</span><p>Good answers show their work.</p><small>Source cards will appear here after your first question.</small></div>
          </>
        )}
      </aside>

      {pendingFile && (
        <div className="modal-backdrop" role="presentation">
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
            <button className="modal-close" type="button" onClick={() => setPendingFile(null)} aria-label="Close">×</button>
            <span className="modal-icon">DOC</span>
            <p className="eyebrow">ADD TO YOUR LIBRARY</p>
            <h2 id="upload-title">Index this document?</h2>
            <p className="modal-copy">Pustak AI will extract its text, create searchable passages, and add them to your local knowledge base.</p>
            <div className="selected-file"><BookIcon /><span><strong>{pendingFile.name}</strong><small>{(pendingFile.size / 1024 / 1024).toFixed(1)} MB</small></span></div>
            <div className="modal-actions"><button type="button" className="secondary" onClick={() => setPendingFile(null)} disabled={busyAction === 'upload'}>Cancel</button><button type="button" className="primary" onClick={() => void uploadDocument()} disabled={busyAction === 'upload'}>{busyAction === 'upload' ? 'Indexing…' : 'Upload & index'}</button></div>
          </section>
        </div>
      )}

      {youtubeOpen && (
        <div className="modal-backdrop" role="presentation">
          <form className="modal" role="dialog" aria-modal="true" aria-labelledby="youtube-title" onSubmit={indexYoutube}>
            <button className="modal-close" type="button" onClick={() => setYoutubeOpen(false)} aria-label="Close">×</button>
            <span className="modal-icon video">▶</span>
            <p className="eyebrow">ADD TO YOUR LIBRARY</p>
            <h2 id="youtube-title">Index a YouTube source</h2>
            <p className="modal-copy">Paste a public video or playlist URL. Available captions will become searchable passages.</p>
            <label className="field-label" htmlFor="youtube-url">YouTube URL</label>
            <input id="youtube-url" type="url" required placeholder="https://www.youtube.com/watch?v=…" value={youtubeUrl} onChange={(event) => setYoutubeUrl(event.target.value)} />
            <div className="modal-actions"><button type="button" className="secondary" onClick={() => setYoutubeOpen(false)} disabled={busyAction === 'youtube'}>Cancel</button><button type="submit" className="primary" disabled={busyAction === 'youtube' || !youtubeUrl.trim()}>{busyAction === 'youtube' ? 'Indexing…' : 'Index source'}</button></div>
          </form>
        </div>
      )}

      {notice && <div className={`toast ${notice.tone}`} role="status"><span>{notice.tone === 'success' ? '✓' : '!'}</span>{notice.text}<button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">×</button></div>}
    </main>
  );
}
