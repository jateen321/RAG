# Gyaan Sarthi Frontend

The frontend is a Vinext/React interface for the Gyaan Sarthi FastAPI service. It
supports document and YouTube ingestion, source-linked chat, conversation
history, prompt editing, image attachments, explicit web search, and generated
study visuals.

## Local development

From the repository root, the recommended launcher starts and supervises both
the frontend and backend:

```bash
npm --prefix frontend ci
.venv/bin/python dev.py
```

Then open <http://localhost:3000>. The FastAPI documentation is available at
<http://127.0.0.1:8000/docs>.

To run only the frontend:

```bash
cp frontend/.env.example frontend/.env.local
npm --prefix frontend run dev -- --host 127.0.0.1
```

`NEXT_PUBLIC_RAG_API_URL` selects the backend URL. It defaults to
`http://127.0.0.1:8000` when unset. `NEXT_PUBLIC_SITE_URL` is used for page
metadata and defaults to `http://localhost:3000`.

## Verification

```bash
npm --prefix frontend run lint
npm --prefix frontend run build
```

The production build is emitted by Vinext. The Python API and its persistent
ChromaDB/conversation data remain separate services; deployment must route the
browser to a reachable API and configure that origin in `RAG_ALLOWED_ORIGINS`.

## Important files

- `app/chat-workspace.tsx` owns chat state, uploads, modes, conversations, and
  API calls.
- `app/answer-markdown.tsx` renders answer Markdown and interactive citations.
- `app/globals.css` defines the responsive visual system.
- `app/layout.tsx` defines page metadata and social sharing metadata.
- `.env.example` documents public browser configuration.

Do not place API secrets in frontend environment files. Variables prefixed with
`NEXT_PUBLIC_` are included in browser-visible code.
