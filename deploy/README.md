# AWS EC2 VM deployment

The app runs as a single-user Docker Compose stack on a persistent EC2 VM. It
uses local ChromaDB and SQLite state, Tesseract by default, and Gemini Developer
API for model requests. OCR runs locally or through the Gemini Developer API.

## Prepare the EC2 instance

Create or use an Ubuntu EC2 instance with enough memory for ChromaDB, the API,
frontend, Redis, and indexing. Attach persistent EBS storage with enough room
for the repository, Docker images, ChromaDB, and uploaded files. Install Docker
Engine and the Docker Compose plugin. Mount or create `/opt/gyaan-sarthi` for
the application and its persistent data.

Configure the instance security group to allow SSH only from your IP. Because
the app has no sign-in, restrict app access to your IP as well. The Caddyfile
uses automatic public certificate validation, which requires the certificate
authority to reach the validation port. If ports 80/443 are IP-restricted,
configure TLS with DNS validation or a pre-provisioned certificate; otherwise
publicly reachable ports make the unauthenticated app accessible to others.
Point the `APP_DOMAIN` DNS A record to the instance's public IP; `API_DOMAIN`
may point to the same address for compatibility checks.

## Copy the app and data

From a trusted machine that has the repository and the indexed data, set these
shell variables for your own instance, then copy the files over SSH:

```bash
export EC2_HOST=your-instance-public-dns
export EC2_USER=ubuntu
export SSH_KEY=~/.ssh/your-ec2-key.pem
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" \
  'sudo mkdir -p /opt/gyaan-sarthi && sudo chown ubuntu:ubuntu /opt/gyaan-sarthi'
git clone https://github.com/jateen321/RAG.git /tmp/gyaan-sarthi-source
rsync -az --exclude .git --exclude .venv --exclude node_modules \
  -e "ssh -i $SSH_KEY" /tmp/gyaan-sarthi-source/ \
  "$EC2_USER@$EC2_HOST:/opt/gyaan-sarthi/"
rsync -az -e "ssh -i $SSH_KEY" chroma_db data \
  "$EC2_USER@$EC2_HOST:/opt/gyaan-sarthi/"
```

Do not copy local `.env` files or credentials. On the instance, create
`.env.production` from the example and set the real Gemini key and domains:

```bash
cd /opt/gyaan-sarthi
cp .env.production.example .env.production
chmod 600 .env.production
```

## Start and verify

Build the app on the instance and start the Compose stack:

```bash
cd /opt/gyaan-sarthi
docker compose --env-file .env.production -f docker-compose.production.yml up -d --build
docker compose --env-file .env.production -f docker-compose.production.yml ps
curl -fsS "https://your-app-domain.example/api/health"
```

Keep the populated `.env.production` file on the instance only. The single-user
app has no sign-in boundary; restrict the security group to your IP.

## Backups and recovery

Take regular EBS snapshots and verify that you can restore one. Back up
`chroma_db/`, `data/`, and the SQLite conversation database before indexing
large document batches. Restore the data and restart the same Compose stack to
recover the instance.

GitHub Actions runs CI checks only; deployment is manual. No cloud deployment
credentials are configured in the workflow. The Compose configuration is
[`../docker-compose.production.yml`](../docker-compose.production.yml), and the
optional image-based release/rollback helper is [`release.sh`](release.sh).
