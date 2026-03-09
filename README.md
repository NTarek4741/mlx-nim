# mlx-nim

> **Early Stage Project** — Core APIs are functional and under active development. Expect breaking changes between releases.

A local inference server for Apple Silicon Macs. mlx-nim runs language and vision models entirely on-device using Apple's [MLX](https://github.com/ml-explore/mlx) framework, exposing a unified HTTP API that multiple clients and agents can connect to simultaneously — no cloud, no telemetry, no data leaving your machine.

The server speaks three API formats at once. Whatever your client already uses — OpenAI-compatible, native Anthropic, or a custom `/api` surface — it connects without modification.

---

## Architecture

```
Clients (Claude Code, OpenCode, Mistral Vibe, n8n, curl, ...)
            │
            ▼
    FastAPI Server :1234
    ┌──────────────────────────────────────────┐
    │  /api/*         Custom chat/mgmt API     │
    │  /v1/chat/*     OpenAI-compatible        │
    │  /v1/messages   Anthropic-compatible     │
    └──────────────────────────────────────────┘
            │
            ▼
    MLX Inference Engine
    (mlx-lm / mlx-vlm, Apple Silicon GPU)
            │
            ▼
    Local Model Store  ./models/
```

Models are loaded on first request and cached in memory. Subsequent requests to the same model incur no reload latency.

---

## Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions — streaming, tool calling, vision | Working |
| `POST` | `/v1/messages` | Anthropic-compatible messages — streaming, tool use, thinking blocks, vision | Working |
| `POST` | `/api/chat` | Chat endpoint — streaming, tool calling, vision, structured output (JSON schema) | Working |
| `GET` | `/v1/models` | List all models available in `./models/` | Working |
| `GET` | `/api/ps` | List currently loaded model | Working |
| `POST` | `/api/pull` | Download a model from Hugging Face Hub | Working |
| `POST` | `/api/create` | Convert and quantize a Hugging Face model to MLX format | Working |
| `DELETE` | `/api/delete` | Remove a model from disk | Working |
| `DELETE` | `/api/clear-huggingface-cache` | Clear the Hugging Face cache directory | Working |
| `GET` | `/api/version` | API version | Working |
| `POST` | `/api/embeddings` | Embeddings generation | Placeholder — not yet implemented |

### Feature Matrix

| Feature | `/v1/chat/completions` | `/v1/messages` | `/api/chat` |
|---------|:---------------------:|:--------------:|:-----------:|
| Streaming | Yes | Yes | Yes |
| Tool / function calling | Yes | Yes | Yes |
| Vision (image inputs) | Yes | Yes | Yes |
| Structured output (JSON schema) | — | — | Yes |
| Speculative decoding | Yes | Yes | Yes |
| KV cache quantization | Yes | Yes | Yes |
| Log probabilities | Yes | — | — |

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

---

## Installation

```bash
git clone https://github.com/your-org/mlx-nim
cd mlx-nim
uv sync
source .venv/bin/activate
```

---

## Running the Server

```bash
uvicorn api.api:app --host 0.0.0.0 --port 1234
```

The server listens at `http://localhost:1234`. Use `--host 0.0.0.0` to allow connections from other devices on your local network.

### Pull a model

```bash
curl -X POST http://localhost:1234/api/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}'
```

Any MLX-format model directory placed in `./models/` is automatically available without a pull. Browse available models at [huggingface.co/mlx-community](https://huggingface.co/mlx-community).

---

## Connecting Clients

### Claude Code

Claude Code connects via the Anthropic-compatible `/v1/messages` endpoint. Set the following environment variables before starting the CLI:

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=local

claude --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

> Coding agents consume context aggressively. Use a model with at least 25k context length — 4-bit quantized 7B+ models are a practical starting point.

---

### OpenCode

OpenCode supports custom providers via its JSON config file. Add a provider block pointing to mlx-nim's OpenAI-compatible `/v1` endpoint.

Edit `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-nim": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mlx-nim",
      "options": {
        "baseURL": "http://localhost:1234/v1"
      },
      "models": {
        "mlx-community/Llama-3.2-3B-Instruct-4bit": {
          "name": "mlx-community/Llama-3.2-3B-Instruct-4bit"
        }
      }
    }
  }
}
```

Restart OpenCode to apply the config, then select `mlx-nim` from the provider list.

> OpenCode benefits from longer contexts. Use a model with at least 64k context length where possible.

---

### Mistral Vibe

Mistral Vibe supports local servers via its `/config` command. Once mlx-nim is running, open Vibe and run `/config`, then set the server to `http://localhost:1234` and select your loaded model.

For a persistent local preset, create a model preset in Vibe's configuration pointing to your mlx-nim instance. This lets you switch between the Mistral API and your local server without re-entering connection details each session.

> Vibe defaults to port 8080 for local servers. If you keep mlx-nim on port 1234, update the port in Vibe's config accordingly.

---

### n8n

Connect n8n workflows to mlx-nim using an **HTTP Request** node targeting the OpenAI-compatible endpoint.

**Node configuration:**

| Field | Value |
|-------|-------|
| Method | `POST` |
| URL | `http://localhost:1234/v1/chat/completions` |
| Content-Type | `application/json` |

**Request body:**

```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "{{ $json.prompt }}" }
  ],
  "stream": false
}
```

Set `"stream": true` for streaming workflows and configure your n8n flow to handle chunked server-sent events.

---

## API Reference

### OpenAI-compatible chat completion

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain KV cache quantization in one paragraph."}
    ],
    "temperature": 0.7,
    "stream": false
  }'
```

### Anthropic-compatible messages

```bash
curl http://localhost:1234/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Explain KV cache quantization in one paragraph."}
    ]
  }'
```

### List available models

```bash
curl http://localhost:1234/v1/models
```

### Check loaded models

```bash
curl http://localhost:1234/api/ps
```

---

## Generation Options

All chat endpoints accept the following optional parameters for controlling generation:

| Parameter | Description |
|-----------|-------------|
| `temperature` | Sampling temperature (default: 1.0) |
| `top_p` | Nucleus sampling probability |
| `top_k` | Top-k sampling |
| `min_p` | Minimum probability threshold |
| `num_predict` | Maximum tokens to generate |
| `stop` | List of stop sequences |
| `seed` | RNG seed for reproducibility |
| `repetition_penalty` | Penalize repeated tokens |
| `kv_bits` | KV cache quantization bits (3–8) |
| `num_draft_tokens` | Tokens per step for speculative decoding |

---

## Built On

mlx-nim is built on top of the following open source projects:

- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — Language model inference on MLX
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language model support
- [mlx-community](https://huggingface.co/mlx-community) — Pre-quantized MLX model weights
- [LM Studio](https://lmstudio.ai) — This project uses MLX Engine for inference 
- [FastAPI](https://fastapi.tiangolo.com) — HTTP server framework

