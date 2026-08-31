# Google Cloud VM deployment

This deployment keeps the current local-state design intact: ChromaDB, SQLite,
uploaded documents, and generated images live on a persistent VM disk. It runs
the FastAPI backend, Next/Vinext frontend, Redis rate limiter, and Caddy HTTPS
proxy with Docker Compose.

## One-time Google Cloud setup

Use a billing-enabled Google Cloud project. The recommended starting size is an
`e2-medium` VM in `asia-south1` with at least a 30 GB standard persistent disk.
The Compute Engine free tier is not sufficient for this application's current
~1.4 GB corpus plus runtime headroom in every region.

```bash
gcloud auth login
gcloud config set project gyaan-sarthi
gcloud services enable compute.googleapis.com secretmanager.googleapis.com
gcloud compute instances create gyaan-sarthi \
  --zone=asia-south1-a \
  --machine-type=e2-medium \
  --boot-disk-size=30GB \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --tags=http-server,https-server
gcloud compute firewall-rules create allow-web \
  --allow=tcp:80,tcp:443 --target-tags=http-server,https-server
```

Point two DNS A records at the VM's external IP: `APP_DOMAIN` for the frontend
and `API_DOMAIN` for the API. Add both hostnames to Firebase Authentication's
Authorized domains. Caddy obtains and renews HTTPS certificates automatically.

## Transfer and start

Install Docker on the VM, clone the repository, and copy the ignored runtime
state from a trusted local backup. Do not copy `.env`, service-account JSON, or
the local `.venv`.

```bash
gcloud compute ssh gyaan-sarthi --zone=asia-south1-a --command='sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin && sudo mkdir -p /opt/gyaan-sarthi && sudo chown "$USER":"$USER" /opt/gyaan-sarthi'
git clone https://github.com/jateen321/RAG.git /tmp/gyaan-sarthi-source
gcloud compute scp --recurse /tmp/gyaan-sarthi-source/. gyaan-sarthi:/opt/gyaan-sarthi/ --zone=asia-south1-a
gcloud compute scp --recurse chroma_db data gyaan-sarthi:/opt/gyaan-sarthi/ --zone=asia-south1-a
```

On the VM, copy `.env.production.example` to `.env.production`, fill in the
Firebase public configuration and backend secret, then start the stack:

```bash
cd /opt/gyaan-sarthi
cp .env.production.example .env.production
docker compose -f docker-compose.production.yml up -d --build
curl -fsS "https://${API_DOMAIN}/health"
```

Use Secret Manager for `GEMINI_API_KEY` and any other sensitive values. Grant the
VM service account only `roles/secretmanager.secretAccessor`; never commit a
service-account key. Before opening the app publicly, verify that `/health`
reports 21,499 shared chunks, guests can ask questions, and only the Firebase
user with the `admin` custom claim can ingest documents.

## Backups and rollback

Schedule daily Compute Engine disk snapshots and test restoring one. Before a
corpus update, snapshot the disk or copy `chroma_db/`, `data/`, and the SQLite
conversation database to a separate backup location. Roll back by stopping the
stack, restoring the snapshot, and starting the same Compose revision.

For higher traffic, migrate ChromaDB and SQLite to managed services before
moving to Cloud Run; Cloud Run's writable filesystem is disposable and does not
preserve these files between instances.
