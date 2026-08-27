'use client';

import { ChangeEvent, FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://127.0.0.1:8000';

type DocumentInfo = {
  source: string;
  chunks: number;
  pages?: number;
  source_type?: string;
  source_url?: string;
};

type HealthResponse = {
  status: string;
  total_chunks: number;
  documents: DocumentInfo[];
};

type Source = {
  chunk_id?: string;
  page?: number;
  source: string;
  distance: number;
  preview: string;
  source_type?: string;
  timestamp?: string;
  timestamp_url?: string;
  source_url?: string;
  video_title?: string;
};

type Passage = {
  chunk_id: string;
  source: string;
  text: string;
  page_number?: number;
  chunk_index?: number;
  source_type?: string;
  start_seconds?: number;
  end_seconds?: number;
};

type PassageViewer = {
  source: Source;
  passage?: Passage;
  loading: boolean;
  error?: string;
};

type AskResponse = {
  answer: string;
  sources: Source[];
  timings?: { total_s?: number };
  conversation_id: string;
};

type Conversation = {
  id: string;
  question: string;
  answer?: string;
  sources?: Source[];
  totalSeconds?: number;
  pending?: boolean;
  error?: string;
};

type ConversationSummary = {
  id: string;
  title: string;
  exchange_count: number;
  created_at: string;
  updated_at: string;
};

type ConversationDetail = ConversationSummary & {
  exchanges: Array<{
    id: string;
    question: string;
    answer: string;
    sources: Source[];
    total_seconds?: number;
  }>;
};

type Notice = { tone: 'success' | 'error'; text: string } | null;

type PendingUpload = {
  files: File[];
  folderName?: string;
  skippedNested: number;
  skippedUnsupported: number;
  skippedOversize: number;
};

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
    const message = payload !== null && typeof payload === 'object' && 'detail' in payload && typeof payload.detail === 'string'
      ? payload.detail
      : `Request failed (${response.status}).`;
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

function sourceHref(source: Source) {
  if (source.timestamp_url) return source.timestamp_url;
  if (source.source_url) return source.source_url;
  if (!['pdf', 'text', 'markdown'].includes(source.source_type || '')) return null;
  const encodedPath = source.source.split('/').map(encodeURIComponent).join('/');
  const pageAnchor = source.source_type === 'pdf' && source.page ? `#page=${source.page}` : '';
  return `${API_URL}/documents/${encodedPath}${pageAnchor}`;
}

function documentHref(document: DocumentInfo) {
  if (document.source_type === 'youtube') return document.source_url || null;
  if (!['pdf', 'text', 'markdown'].includes(document.source_type || '')) return null;
  const encodedPath = document.source.split('/').map(encodeURIComponent).join('/');
  return `${API_URL}/documents/${encodedPath}`;
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
  const [conversationHistory, setConversationHistory] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [activeSources, setActiveSources] = useState<Source[]>([]);
  const [passageViewer, setPassageViewer] = useState<PassageViewer | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileLibraryOpen, setMobileLibraryOpen] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [youtubeOpen, setYoutubeOpen] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [busyAction, setBusyAction] = useState<'upload' | 'youtube' | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
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

  const loadConversation = useCallback(async (conversationId: string) => {
    setLoadingConversation(true);
    try {
      const detail = await requestJson<ConversationDetail>(`/conversations/${conversationId}`);
      setActiveConversationId(detail.id);
      setConversations(detail.exchanges.map((exchange) => ({
        id: exchange.id,
        question: exchange.question,
        answer: exchange.answer,
        sources: exchange.sources,
        totalSeconds: exchange.total_seconds,
      })));
      setActiveSources(detail.exchanges.at(-1)?.sources || []);
      setMobileLibraryOpen(false);
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not load that conversation.' });
    } finally {
      setLoadingConversation(false);
    }
  }, []);

  const refreshConversationHistory = useCallback(async (openMostRecent = false) => {
    try {
      const data = await requestJson<{ conversations: ConversationSummary[] }>('/conversations');
      setConversationHistory(data.conversations);
      if (openMostRecent && data.conversations[0]) {
        await loadConversation(data.conversations[0].id);
      }
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not load conversation history.' });
    }
  }, [loadConversation]);

  useEffect(() => { void refreshHealth(); }, [refreshHealth]);
  useEffect(() => { void refreshConversationHistory(true); }, [refreshConversationHistory]);
  useEffect(() => { threadEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [conversations]);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function ask(nextQuestion = question) {
    const cleanQuestion = nextQuestion.trim();
    if (!cleanQuestion || conversations.some((message) => message.pending)) return;

    const id = crypto.randomUUID();
    setQuestion('');
    setConversations((current) => [...current, { id, question: cleanQuestion, pending: true }]);
    try {
      const data = await requestJson<AskResponse>('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: cleanQuestion,
          conversation_id: activeConversationId || undefined,
        }),
      });
      setConversations((current) => current.map((message) => message.id === id ? {
        ...message,
        pending: false,
        answer: data.answer,
        sources: data.sources,
        totalSeconds: data.timings?.total_s,
      } : message));
      setActiveConversationId(data.conversation_id);
      setActiveSources(data.sources || []);
      await refreshConversationHistory();
    } catch (error) {
      setConversations((current) => current.map((message) => message.id === id ? {
        ...message,
        pending: false,
        error: error instanceof Error ? error.message : 'The question could not be answered.',
      } : message));
    }
  }

  function startNewConversation() {
    setActiveConversationId(null);
    setConversations([]);
    setActiveSources([]);
    setQuestion('');
    setMobileLibraryOpen(false);
  }

  async function removeConversation(conversationId: string) {
    const conversation = conversationHistory.find((item) => item.id === conversationId);
    if (!window.confirm(`Delete “${conversation?.title || 'this conversation'}”?`)) return;
    try {
      await requestJson(`/conversations/${conversationId}`, { method: 'DELETE' });
      if (activeConversationId === conversationId) startNewConversation();
      await refreshConversationHistory();
    } catch (error) {
      setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not delete the conversation.' });
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
    setPendingUpload({ files: [file], skippedNested: 0, skippedUnsupported: 0, skippedOversize: 0 });
  }

  function chooseFolder(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) return;

    const relativePath = (file: File) => file.webkitRelativePath || file.name;
    const firstPathParts = relativePath(selectedFiles[0]).split('/');
    const folderName = firstPathParts.length > 1 ? firstPathParts[0] : 'Selected folder';
    const topLevelFiles = selectedFiles.filter((file) => relativePath(file).split('/').length <= 2);
    const supportedFiles = topLevelFiles.filter((file) => ['.pdf', '.txt', '.md'].some((extension) => file.name.toLowerCase().endsWith(extension)));
    const files = supportedFiles.filter((file) => file.size <= 500 * 1024 * 1024);
    const skippedNested = selectedFiles.length - topLevelFiles.length;
    const skippedUnsupported = topLevelFiles.length - supportedFiles.length;
    const skippedOversize = supportedFiles.length - files.length;

    if (!files.length) {
      setNotice({ tone: 'error', text: `No eligible top-level PDF, TXT, or Markdown files were found in ${folderName}.` });
      event.target.value = '';
      return;
    }

    setPendingUpload({ files, folderName, skippedNested, skippedUnsupported, skippedOversize });
  }

  function closeUploadDialog() {
    setPendingUpload(null);
    setUploadProgress(0);
    if (fileInput.current) fileInput.current.value = '';
    if (folderInput.current) folderInput.current.value = '';
  }

  async function uploadDocument() {
    if (!pendingUpload) return;
    setBusyAction('upload');
    setUploadProgress(0);
    const indexed: string[] = [];
    const failures: string[] = [];

    for (const [index, file] of pendingUpload.files.entries()) {
      const body = new FormData();
      body.append('file', file);
      if (pendingUpload.folderName && file.webkitRelativePath) body.append('relative_path', file.webkitRelativePath);
      try {
        const result = await requestJson<{ source: string; pages_with_text: number; chunks_indexed: number }>('/upload', {
          method: 'POST',
          body,
        });
        indexed.push(result.source);
      } catch (error) {
        failures.push(`${file.name}: ${error instanceof Error ? error.message : 'could not be indexed'}`);
      }
      setUploadProgress(index + 1);
    }

    if (indexed.length) await refreshHealth();
    if (failures.length) {
      setNotice({ tone: 'error', text: `Indexed ${indexed.length} of ${pendingUpload.files.length}. ${failures.length} failed. ${failures[0]}` });
    } else if (pendingUpload.folderName) {
      setNotice({ tone: 'success', text: `${indexed.length} documents from ${pendingUpload.folderName} were indexed.` });
    } else {
      setNotice({ tone: 'success', text: `${indexed[0]} indexed successfully.` });
    }
    closeUploadDialog();
    setBusyAction(null);
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

  async function viewPassage(source: Source) {
    setPassageViewer({ source, loading: true });
    try {
      const passagePath = source.chunk_id
        ? `/passages/${encodeURIComponent(source.chunk_id)}?source=${encodeURIComponent(source.source)}`
        : `/passages/resolve-legacy?source=${encodeURIComponent(source.source)}&page=${encodeURIComponent(String(source.page || 0))}&preview=${encodeURIComponent(source.preview)}`;
      const passage = await requestJson<Passage>(
        passagePath,
      );
      setPassageViewer({ source, passage, loading: false });
    } catch (error) {
      setPassageViewer({
        source,
        loading: false,
        error: error instanceof Error ? error.message : 'The cited passage could not be loaded.',
      });
    }
  }

  return (
    <main className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <aside id="library-sidebar" className={`library-panel ${mobileLibraryOpen ? 'mobile-open' : ''}`}>
        <div className="mobile-panel-head">
          <div className="brand">
            <span className="brand-mark">स</span>
            <div><strong>Sarthi AI</strong><span>Grounded study companion</span></div>
          </div>
          <button className="close-panel" type="button" onClick={() => setMobileLibraryOpen(false)} aria-label="Close library">×</button>
        </div>

        <button className="new-chat" type="button" onClick={startNewConversation}>
          <span aria-hidden="true">＋</span> New conversation
        </button>

        <div className="history-heading">Recent conversations</div>
        <div className="conversation-history" aria-label="Saved conversations">
          {conversationHistory.map((conversation) => (
            <div className={`history-item ${activeConversationId === conversation.id ? 'active' : ''}`} key={conversation.id}>
              <button className="history-open" type="button" onClick={() => void loadConversation(conversation.id)} disabled={loadingConversation} title={conversation.title}>
                <span>{conversation.title}</span>
                <small>{conversation.exchange_count} exchange{conversation.exchange_count === 1 ? '' : 's'}</small>
              </button>
              <button className="history-delete" type="button" onClick={() => void removeConversation(conversation.id)} aria-label={`Delete ${conversation.title}`}>×</button>
            </div>
          ))}
          {!conversationHistory.length && <p className="empty-history">Your conversations will appear here.</p>}
        </div>

        <div className="library-heading">
          <span>Your library</span>
          <span className="library-count">{health?.documents.length ?? '—'}</span>
        </div>

        <div className="document-list">
          {!health && !healthError && <div className="library-loading"><span /><span /><span /></div>}
          {healthError && <button className="inline-error" type="button" onClick={() => void refreshHealth()}>{healthError} <strong>Retry</strong></button>}
          {health?.documents.map((document) => {
            const href = documentHref(document);
            const content = <>
              <BookIcon youtube={document.source_type === 'youtube'} />
              <span>
                <strong>{document.source}</strong>
                <small>{document.pages ? `${document.pages} pages · ` : ''}{document.chunks} passages</small>
              </span>
            </>;

            return href ? (
              <a className="document-card" href={href} key={document.source} title={`Open ${document.source}`} target="_blank" rel="noopener noreferrer">
                {content}
              </a>
            ) : (
              <div className="document-card" key={document.source} title={`${document.source} cannot be opened because its source URL is unavailable.`}>
                {content}
              </div>
            );
          })}
          {health && health.documents.length === 0 && <p className="empty-library">Your library is empty. Add a document or YouTube source to begin.</p>}
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
          <div className="header-actions" role="group" aria-label="Add sources">
            <button type="button" onClick={() => fileInput.current?.click()} aria-label="Add a document" title="Add a document">
              <span aria-hidden="true">↑</span><span className="header-action-label">Add a document</span>
            </button>
            <button type="button" onClick={() => folderInput.current?.click()} aria-label="Add a folder" title="Add a folder">
              <span aria-hidden="true">▤</span><span className="header-action-label">Add a folder</span>
            </button>
            <button type="button" onClick={() => setYoutubeOpen(true)} aria-label="Add YouTube" title="Add YouTube">
              <span aria-hidden="true">▶</span><span className="header-action-label">Add YouTube</span>
            </button>
          </div>
          <input ref={fileInput} className="visually-hidden" type="file" accept="application/pdf,text/plain,text/markdown,.pdf,.txt,.md" onChange={chooseFile} tabIndex={-1} aria-hidden="true" />
          <input
            ref={(node) => {
              folderInput.current = node;
              if (node) {
                node.setAttribute('webkitdirectory', '');
                node.setAttribute('directory', '');
              }
            }}
            id="folder-upload"
            className="visually-hidden"
            type="file"
            multiple
            onChange={chooseFolder}
            tabIndex={-1}
            aria-hidden="true"
          />
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
                  <div className="assistant-badge">स</div>
                  <div className="answer-content">
                    <div className="answer-heading"><strong>Sarthi AI</strong>{message.totalSeconds != null && <small>{message.totalSeconds.toFixed(1)}s · {message.sources?.length ?? 0} sources</small>}</div>
                    {message.pending && <LoadingAnswer />}
                    {message.error && <div className="answer-error"><strong>I couldn’t answer that.</strong><p>{message.error}</p><button type="button" onClick={() => void ask(message.question)}>Try again</button></div>}
                    {message.answer && <div className="answer-copy">{message.answer}</div>}
                    {!!message.sources?.length && (
                      <>
                        <button className="show-sources" type="button" onClick={() => setActiveSources(message.sources || [])}>
                          View {message.sources.length} supporting sources <span aria-hidden="true">→</span>
                        </button>
                        <div className="inline-sources">
                          {message.sources.map((source, index) => {
                            const href = sourceHref(source);
                            const content = <><strong>{index + 1}. {locationLabel(source)}</strong><span>{source.source}</span></>;
                            return source.preview
                              ? <button type="button" onClick={() => void viewPassage(source)} aria-label={`View cited passage from ${source.source}`} key={`${source.source}-${index}`}>{content}</button>
                              : href
                                ? <a href={href} target="_blank" rel="noreferrer" aria-label={`Open ${source.source} at ${locationLabel(source)}`} key={`${source.source}-${index}`}>{content}</a>
                                : <div key={`${source.source}-${index}`}>{content}</div>;
                          })}
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
                const href = sourceHref(source);
                const content = (
                  <>
                    <div className="source-card-head"><span>{String(index + 1).padStart(2, '0')}</span><strong>{locationLabel(source)}</strong></div>
                    <h3>{source.video_title || source.source}</h3>
                    <p>{source.preview}</p>
                    <small>{source.source_type === 'youtube' ? 'YouTube transcript' : source.source}</small>
                  </>
                );
                return source.preview
                  ? <button className="source-card" type="button" onClick={() => void viewPassage(source)} aria-label={`View cited passage from ${source.source}`} key={`${source.source}-${index}`}>{content}</button>
                  : href
                    ? <a className="source-card" href={href} target="_blank" rel="noreferrer" aria-label={`Open ${source.source}`} key={`${source.source}-${index}`}>{content}</a>
                    : <article className="source-card" key={`${source.source}-${index}`}>{content}</article>;
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

      {passageViewer && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setPassageViewer(null);
        }}>
          <section className="modal passage-modal" role="dialog" aria-modal="true" aria-labelledby="passage-title">
            <button className="modal-close" type="button" onClick={() => setPassageViewer(null)} aria-label="Close passage">×</button>
            <p className="eyebrow">RETRIEVED EVIDENCE</p>
            <h2 id="passage-title">{locationLabel(passageViewer.source)}</h2>
            {sourceHref(passageViewer.source) ? (
              <a
                className="passage-source passage-source-link"
                href={sourceHref(passageViewer.source) || undefined}
                target="_blank"
                rel="noreferrer"
                aria-label={`Open ${passageViewer.source.video_title || passageViewer.source.source} at ${locationLabel(passageViewer.source)}`}
              >
                {passageViewer.source.video_title || passageViewer.source.source}
                <span aria-hidden="true">↗</span>
              </a>
            ) : (
              <p className="passage-source">{passageViewer.source.video_title || passageViewer.source.source}</p>
            )}

            {passageViewer.loading && <div className="passage-loading" role="status"><span /><span /><span /></div>}
            {passageViewer.error && <div className="passage-error"><strong>Passage unavailable</strong><p>{passageViewer.error}</p></div>}
            {passageViewer.passage && (
              <blockquote className="passage-text">{passageViewer.passage.text}</blockquote>
            )}

            <div className="modal-actions">
              <button type="button" className="secondary" onClick={() => setPassageViewer(null)}>Close</button>
              {sourceHref(passageViewer.source) && (
                <a className="primary" href={sourceHref(passageViewer.source) || undefined} target="_blank" rel="noreferrer">
                  Open original at {locationLabel(passageViewer.source)}
                </a>
              )}
            </div>
          </section>
        </div>
      )}

      {pendingUpload && (
        <div className="modal-backdrop" role="presentation">
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
            <button className="modal-close" type="button" onClick={closeUploadDialog} aria-label="Close">×</button>
            <span className="modal-icon">{pendingUpload.folderName ? 'DIR' : 'DOC'}</span>
            <p className="eyebrow">ADD TO YOUR LIBRARY</p>
            <h2 id="upload-title">Index this {pendingUpload.folderName ? 'folder' : 'document'}?</h2>
            <p className="modal-copy">Sarthi AI will extract the text, create searchable passages, and preserve the selected folder path in your local library.</p>
            <div className="selected-file"><BookIcon /><span><strong>{pendingUpload.folderName || pendingUpload.files[0].name}</strong><small>{pendingUpload.folderName ? `${pendingUpload.files.length} supported top-level document${pendingUpload.files.length === 1 ? '' : 's'}` : `${(pendingUpload.files[0].size / 1024 / 1024).toFixed(1)} MB`}</small></span></div>
            {pendingUpload.folderName && (pendingUpload.skippedNested > 0 || pendingUpload.skippedUnsupported > 0 || pendingUpload.skippedOversize > 0) && (
              <p className="selection-note">Skipped: {pendingUpload.skippedNested} from nested folders, {pendingUpload.skippedUnsupported} unsupported, {pendingUpload.skippedOversize} over 500 MB.</p>
            )}
            <div className="modal-actions"><button type="button" className="secondary" onClick={closeUploadDialog} disabled={busyAction === 'upload'}>Cancel</button><button type="button" className="primary" onClick={() => void uploadDocument()} disabled={busyAction === 'upload'}>{busyAction === 'upload' ? `Indexing ${uploadProgress}/${pendingUpload.files.length}…` : 'Upload & index'}</button></div>
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
