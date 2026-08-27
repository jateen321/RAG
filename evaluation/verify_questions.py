import json, re, sys, chromadb, config
col = chromadb.PersistentClient(path=config.CHROMA_DB_PATH).get_collection(config.COLLECTION_NAME)
norm = lambda s: re.sub(r'\s+', ' ', s).strip()

data = json.load(open(sys.argv[1], encoding='utf-8'))
ok = fail = 0
for q in data:
    if q.get('category') == 'unanswerable':
        # must NOT be well covered; checked separately
        print(f"[skip] {q['id']}: unanswerable"); continue
    hits = []
    for src in q['expected_sources']:
        g = col.get(where={"source_name": src}, limit=5000, include=["documents","metadatas"])
        for doc, md in zip(g['documents'], g['metadatas']):
            if md.get('page_number') in q['expected_pages']:
                hits.append(norm(doc))
    blob = ' '.join(hits)
    ev = norm(q['evidence'])
    found = ev in blob
    # fall back: longest common run check to report *why* it failed
    if found:
        ok += 1; print(f"[ OK ] {q['id']}: evidence found in {len(hits)} chunk(s) at page(s) {q['expected_pages']}")
    else:
        fail += 1
        print(f"[FAIL] {q['id']}: evidence NOT found. chunks at page: {len(hits)}")
        if hits:
            frag = ev[:40]
            print(f"        looking for: {frag!r}")
            print(f"        page text  : {blob[:160]!r}")
print(f"\nverified {ok} ok, {fail} failed, of {len(data)}")
