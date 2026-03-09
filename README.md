# mlx-nim

> **Early Stage Project** — APIs are functional but under active development. Expect breaking changes.

A local, high-performance inference server for Apple Silicon Macs. Built on [MLX](https://github.com/ml-explore/mlx), mlx-nim lets a single machine serve language and vision models to multiple clients simultaneously over HTTP — no cloud required, no data leaving your machine.

The server exposes three API layers simultaneously, so coding agents, automation tools, and custom clients can all connect using whatever API format they already speak: Ollama-style, OpenAI-compatible, or native Anthropic.

---

## Architecture

```
Clients (Claude Code, OpenCode, LM Studio, n8n, curl, ...)
            │
            ▼
    FastAPI Server :1234
    ┌─────────────────────────────────────┐
    │  /api/*        Ollama-style API     │
    │  /v1/chat/*    OpenAI-compatible    │
    │  /v1/messages  Anthropic-compatible │
    └─────────────────────────────────────┘
            │
            ▼
    MLX Inference Engine
    (mlx-lm / mlx-vlm, Apple Silicon GPU)
            │
            ▼
    Local Model Store  ./models/
```

Models are cached in memory between requests. The first request loads the model; subsequent requests reuse it with no reload overhead.

---

## Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/api/chat` | Ollama-compatible chat — streaming, tool calling, vision, structured output | Working |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions — streaming, tool calling | Working |
| `POST` | `/v1/messages` | Anthropic-compatible messages — streaming, tool use, thinking blocks, vision | Working |
| `GET` | `/v1/models` | List all models available in `./models/` | Working |
| `GET` | `/api/ps` | List currently loaded models and memory usage | Working |
| `POST` | `/api/pull` | Download a model from Hugging Face Hub | Working |
| `POST` | `/api/create` | Convert and quantize a Hugging Face model to MLX format | Working |
| `DELETE` | `/api/delete` | Remove a model from disk | Working |
| `DELETE` | `/api/clear-huggingface-cache` | Clear the Hugging Face cache directory | Working |
| `GET` | `/api/version` | Returns the API version | Working |
| `POST` | `/api/embeddings` | Generate embeddings | Placeholder — not yet implemented |

### Feature Support

| Feature | `/api/chat` | `/v1/chat/completions` | `/v1/messages` |
|---------|:-----------:|:---------------------:|:--------------:|
| Streaming | Yes | Yes | Yes |
| Tool / function calling | Yes | Yes | Yes |
| Vision (image inputs) | Yes | Yes | Yes |
| Structured output (JSON schema) | Yes | Yes | — |
| Speculative decoding | Yes | Yes | Yes |
| KV cache quantization | Yes | Yes | Yes |
| Log probabilities | — | Yes | — |

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

---

## Getting Started

### 1. Install dependencies

```bash
uv sync
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

Or prefix all commands with `uv run`.

### 3. Pull a model

```bash
curl -X POST http://localhost:1234/api/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}'
```

Alternatively, place any MLX-format model directory inside `./models/`.

### 4. Start the server

```bash
uvicorn api.api:app --host 0.0.0.0 --port 1234
```

The server is now accepting requests at `http://localhost:1234`.

---

## Connecting Clients

### Claude Code

Point Claude Code at mlx-nim instead of the Anthropic API. The `/v1/messages` endpoint is fully Anthropic-compatible.

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=local
```

Then run Claude Code against whichever MLX model you have loaded:

```bash
claude --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

> **Tip:** Use a model with at least 25k context length. Coding agents consume context quickly. 4-bit quantized 7B+ models are a good starting point.

---

### OpenCode

OpenCode uses a JSON config file. Add a provider block pointing to mlx-nim's OpenAI-compatible endpoint at `/v1`:

Edit `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-nim": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mlx-nim (local)",
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

Restart OpenCode to load the new config, then select your local model from the provider list.

> Replace the model name with whatever model you have pulled into `./models/`. Use a model with at least 64k context for best results with OpenCode.

---

### LM Studio

LM Studio can act as a client to any OpenAI-compatible server. Point it at mlx-nim by setting the server URL to `http://localhost:1234/v1` in LM Studio's remote server configuration. This lets you use LM Studio's chat UI while mlx-nim handles inference on the Apple Silicon GPU.

Alternatively, if you are already running LM Studio as your inference backend and want to migrate to mlx-nim, simply swap the server URL — the OpenAI-compatible API surface is identical.

---

### n8n

Use an **HTTP Request** node pointed at the OpenAI-compatible endpoint.

| Field | Value |
|-------|-------|
| Method | `POST` |
| URL | `http://localhost:1234/v1/chat/completions` |
| Authentication | None |
| Content-Type | `application/json` |

Example body:

```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "messages": [
    { "role": "user", "content": "{{ $json.prompt }}" }
  ],
  "stream": false
}
```

For streaming workflows, set `"stream": true` and handle the response as a stream in your n8n flow.

---

## Example API Calls

### OpenAI-compatible

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Anthropic-compatible

```bash
curl http://localhost:1234/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### List available models

```bash
curl http://localhost:1234/v1/models
```




