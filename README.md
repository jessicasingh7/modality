# Modality

Fine-tune small language models (SLMs) for enterprise customers and intelligently route traffic to them — instead of paying for GPT-4o on every request.

## What this does

Enterprise customers have narrow, repeated use cases — legal contract review, financial report summarization, customer support triage. A fine-tuned 8B parameter model can match GPT-4o quality on those tasks at 10-50x lower cost.

Modality handles the full lifecycle:

1. **Onboard** a customer and define their domain (e.g. "legal", "finance")
2. **Fine-tune** an SLM on their data via OpenAI or Fireworks APIs
3. **Evaluate** the model automatically before it goes live
4. **Route** incoming requests to the best model — or fall back to GPT-4o when unsure
5. **Track** usage and cost savings per customer

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       CUSTOMER'S APP                         │
│                                                              │
│  client = OpenAI(base_url="https://api.modality.dev/v1",    │
│                  api_key="mod_abc123...")                     │
│  response = client.chat.completions.create(                  │
│      model="auto",    # Modality picks the best model        │
│      messages=[...]                                          │
│  )                                                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     DATA PLANE (port 8000)                   │
│               internet-facing, autoscaled                    │
│                                                              │
│  1. Authenticate request via API key                         │
│  2. Embed the prompt                                         │
│  3. Compare against cached model domain embeddings           │
│  4. Route to best SLM (or fall back to GPT-4o)              │
│  5. Return response + log usage                              │
│                                                              │
│  POST /v1/chat/completions                                   │
│  GET  /health                                                │
└──────────────────────────────────────────────────────────────┘
                       │
          reads from   │   shared database
                       │
┌──────────────────────────────────────────────────────────────┐
│                   CONTROL PLANE (port 8001)                  │
│             internal only, behind VPN                        │
│                                                              │
│  • Onboard customers          POST /customers                │
│  • Issue API keys             POST /customers/:id/api-keys   │
│  • Upload data & fine-tune    POST /finetune                 │
│  • Manage models              POST /models/:id/promote       │
│  • View usage & savings       GET  /customers/:id/usage      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Why two planes?

The data plane handles thousands of requests per second and must respond in <200ms. The control plane handles a few requests per day (uploading data, starting fine-tune jobs). Separating them means:

- A slow fine-tune upload never causes latency spikes on inference
- You scale them independently (20 data plane replicas, 1 control plane replica)
- If the control plane goes down, all customer apps keep working
- Different security posture: data plane is internet-facing, control plane is internal

They share the same codebase and database — they're just different entry points deployed as separate services.

## How customers use it

From the customer's perspective, Modality is a **drop-in replacement for the OpenAI API**. They change two lines:

```python
# Before — calling OpenAI directly ($$$)
client = OpenAI(api_key="sk-...")

# After — calling Modality (routes to their fine-tuned SLM)
client = OpenAI(
    base_url="https://api.modality.dev/v1",
    api_key="mod_abc123..."  # issued via control plane
)

# Same API, same code, lower cost
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Summarize this contract..."}]
)
```

The customer doesn't choose which model to use. Modality's router picks the best fine-tuned model for each request based on what it's about — or falls back to GPT-4o for anything outside the model's domain.

## Running locally

```bash
# 1. Configure API keys
cp .env.example .env
# Edit .env with your OpenAI / Fireworks keys

# 2. Start everything
docker compose up --build

# Data plane:    http://localhost:8000
# Control plane: http://localhost:8001
# Swagger docs:  http://localhost:8000/docs and http://localhost:8001/docs
```

## Deploying to production

### Option A: Docker Compose (single server)

Good for getting started. The `docker-compose.yml` runs both planes + Postgres.

### Option B: Kubernetes / ECS / Cloud Run (recommended)

Deploy each plane as a separate service from the same Docker image:

```bash
# Build the image
docker build --target data-plane -t modality-data-plane .
docker build --target control-plane -t modality-control-plane .
```

**Data plane service:**
- Internet-facing, behind a load balancer
- Autoscale on CPU/request count (start with 2, scale to 20+)
- Set `MODALITY_DATABASE_URL` to your managed Postgres (RDS, Cloud SQL, etc.)
- Health check: `GET /health`

**Control plane service:**
- Internal only — behind VPN, private subnet, or IP-allowlisted
- 1-2 replicas is enough
- Same database connection string
- Health check: `GET /health`

**Database:**
- Use managed Postgres (RDS, Cloud SQL, Neon, Supabase)
- Both planes connect to the same database
- The data plane only reads; the control plane reads and writes

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `MODALITY_DATABASE_URL` | Postgres connection string | `sqlite+aiosqlite:///./modality.db` |
| `MODALITY_OPENAI_API_KEY` | OpenAI API key for embeddings + fine-tuning | — |
| `MODALITY_FIREWORKS_API_KEY` | Fireworks API key (optional) | — |
| `MODALITY_FALLBACK_MODEL` | Large model used when no SLM matches | `gpt-4o` |
| `MODALITY_ROUTER_CONFIDENCE_THRESHOLD` | Minimum similarity score to route to an SLM | `0.7` |
| `MODALITY_EVAL_MIN_SCORE` | Minimum eval score to auto-promote a model | `0.8` |

## How routing works

1. Customer sends a request to `/v1/chat/completions`
2. The router embeds the prompt using `text-embedding-3-small` (fast, cheap)
3. It compares the embedding against every active model's domain embedding using cosine similarity
4. If the best match scores above the confidence threshold (default 0.7), route to that SLM
5. If nothing matches well enough, fall back to GPT-4o

The domain embeddings are generated when a model is fine-tuned, based on the `domain_description` you provide (e.g. "Legal contract analysis, clause extraction, risk assessment for US corporate law"). They're cached in memory on the data plane so routing adds <5ms of latency.

## How fine-tuning works

1. Prepare training data as JSONL in OpenAI chat format:
   ```json
   {"messages": [{"role": "system", "content": "You are a legal assistant."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```

2. Call the control plane:
   ```bash
   curl -X POST http://localhost:8001/finetune \
     -H "Content-Type: application/json" \
     -d '{
       "customer_name": "Acme Legal",
       "domain": "legal",
       "domain_description": "Legal contract analysis, clause extraction, and risk assessment for US corporate law",
       "training_file_path": "/data/acme/train.jsonl",
       "base_model": "gpt-4o-mini-2024-07-18",
       "provider": "openai"
     }'
   ```

3. Modality uploads the data to the provider, starts the fine-tuning job, polls for completion, runs an automated evaluation (LLM-as-judge), and promotes the model into the routing table if it scores above the threshold.

## Project structure

```
modality/
├── config.py                        # Settings and environment variables
├── gateway/
│   ├── data_plane.py                # Customer-facing inference API
│   ├── control_plane.py             # Internal management API
│   ├── auth.py                      # API key authentication
│   └── schemas.py                   # Request/response models
├── router/
│   ├── router.py                    # Core routing logic
│   ├── cache.py                     # In-memory model cache for the data plane
│   └── schemas.py                   # RouteDecision model
├── finetune/
│   ├── pipeline.py                  # Fine-tune orchestration (upload → train → eval → promote)
│   └── data.py                      # JSONL validation and train/eval splitting
├── eval/
│   └── evaluator.py                 # LLM-as-judge evaluation
├── providers/
│   ├── base.py                      # Abstract provider interface
│   ├── openai_provider.py           # OpenAI integration
│   ├── fireworks_provider.py        # Fireworks integration
│   ├── embeddings.py                # Embedding API for the router
│   └── registry.py                  # Provider factory
└── registry/
    ├── database.py                  # Async SQLAlchemy setup
    ├── models.py                    # DB models (Customer, Model, Job, ApiKey, UsageLog)
    └── service.py                   # CRUD operations
```
