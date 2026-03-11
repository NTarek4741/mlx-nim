# api_tests/speed_test.py
#
# Run: python -m api_tests.speed_test [--server {mlx,ollama,both}]
# Prerequisites: pip install ollama openai anthropic
#
# Compares MLX-NIM (port 1234) vs standard Ollama (port 11434).
# Run with only one server active at a time for accurate isolated measurements.
#
#   --server mlx     Only hit MLX-NIM (port 1234). Stop Ollama first.
#   --server ollama  Only hit standard Ollama (port 11434). Stop MLX-NIM first.
#   --server both    Hit both servers (default). Shares unified memory.
#
# Model pairs (same architecture, comparable size):
#   MLX-NIM                                      | Standard Ollama
#   mlx-community/Ministral-3-3B-Instruct-*      | ministral-3:3b
#   qwen/Qwen3.5-4B-MLX-4bit                     | qwen3.5:4b

import argparse
import base64
import time
import traceback
from pathlib import Path

import anthropic
import ollama
import openai

# =============================================================================
# Config
# =============================================================================

MLX_URL    = "http://localhost:1234"
OLLAMA_URL = "http://localhost:11434"

MODEL_PAIRS = [
    ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", "ministral-3:3b"),
    ("qwen/Qwen3.5-4B-MLX-4bit",                        "qwen3.5:4b"),
]

DEMO_DATA = Path(__file__).parent.parent / "demo-data"
CHAMELEON = DEMO_DATA / "chameleon.webp"
TOUCAN    = DEMO_DATA / "toucan.jpeg"

MEDIA_TYPES = {
    ".webp": "image/webp",
    ".jpeg": "image/jpeg",
    ".jpg":  "image/jpeg",
    ".png":  "image/png",
}

# =============================================================================
# Clients
# =============================================================================

mlx_ollama    = ollama.Client(host=MLX_URL)
std_ollama    = ollama.Client(host=OLLAMA_URL)
mlx_openai    = openai.OpenAI(base_url=f"{MLX_URL}/v1", api_key="x")
mlx_anthropic = anthropic.Anthropic(base_url=MLX_URL, api_key="x")

# =============================================================================
# Shared prompts and tools
# =============================================================================

SYSTEM_PROMPT = (
    "You are a concise creative writing assistant. "
    "Respond in no more than two sentences."
)
TEXT_PROMPT   = "Write a one-sentence description of a distant star."
VISION_PROMPT = (
    "You are looking at two images. Briefly describe what is in each image "
    "and name one key visual difference between them."
)
TOOL_PROMPT   = "What is the current weather in Paris? Use the available tool."

WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name, e.g. Paris"},
                "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city name, e.g. Paris"},
            "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}

# =============================================================================
# Helpers
# =============================================================================


def load_image_b64(path: Path) -> tuple[str, str]:
    data       = path.read_bytes()
    b64        = base64.standard_b64encode(data).decode()
    media_type = MEDIA_TYPES[path.suffix.lower()]
    return b64, media_type


def fmt_ollama(resp) -> str:
    """Format timing from an Ollama response object."""
    parts = []
    if getattr(resp, "load_duration", None):
        parts.append(f"load={resp.load_duration / 1e9:.2f}s")
    pd = getattr(resp, "prompt_eval_duration", None)
    pc = getattr(resp, "prompt_eval_count", None)
    if pd and pc:
        parts.append(f"pp={pc / (pd / 1e9):.0f}t/s")
    ed = getattr(resp, "eval_duration", None)
    ec = getattr(resp, "eval_count", None)
    if ed and ec:
        parts.append(f"gen={ec / (ed / 1e9):.0f}t/s")
    return "  ".join(parts) or "—"


def fmt_wall(t0: float, out_tokens: int) -> str:
    """Wall-clock speed for OpenAI/Anthropic endpoints."""
    elapsed = time.time() - t0
    if elapsed > 0 and out_tokens:
        return f"gen={out_tokens / elapsed:.0f}t/s  wall={elapsed:.1f}s"
    return f"wall={elapsed:.1f}s"


# =============================================================================
# Speed benchmarks — text
# =============================================================================


def text_speed(mlx_model: str, std_model: str, server: str = "both") -> dict[str, str]:
    msgs_ollama = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": TEXT_PROMPT},
    ]
    msgs_sdk = [{"role": "user", "content": TEXT_PROMPT}]
    opts = {"temperature": 0.7, "num_predict": 128}

    results: dict[str, str] = {}

    if server in ("mlx", "both"):
        resp = mlx_ollama.chat(model=mlx_model, messages=msgs_ollama, stream=False, options=opts)
        results["mlx/ollama"] = fmt_ollama(resp)

        t0 = time.time()
        resp_oa = mlx_openai.chat.completions.create(
            model=mlx_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *msgs_sdk],
            max_tokens=128, temperature=0.7,
        )
        results["mlx/openai"] = fmt_wall(t0, resp_oa.usage.completion_tokens)

        t0 = time.time()
        resp_an = mlx_anthropic.messages.create(
            model=mlx_model, system=SYSTEM_PROMPT, messages=msgs_sdk,
            max_tokens=128, temperature=0.7,
        )
        results["mlx/anthropic"] = fmt_wall(t0, resp_an.usage.output_tokens)

    if server in ("ollama", "both"):
        resp_std = std_ollama.chat(model=std_model, messages=msgs_ollama, stream=False, options=opts)
        results["std/ollama"] = fmt_ollama(resp_std)

    return results


# =============================================================================
# Speed benchmarks — vision
# =============================================================================


def vision_speed(mlx_model: str, std_model: str, server: str = "both") -> dict[str, str]:
    b64_c, mt_c = load_image_b64(CHAMELEON)
    b64_t, mt_t = load_image_b64(TOUCAN)
    opts = {"temperature": 0.5, "num_predict": 256}

    results: dict[str, str] = {}

    if server in ("mlx", "both"):
        resp = mlx_ollama.chat(
            model=mlx_model, stream=False,
            messages=[{"role": "user", "content": VISION_PROMPT, "images": [b64_c, b64_t]}],
            options=opts,
        )
        results["mlx/ollama"] = fmt_ollama(resp)

        t0 = time.time()
        resp_oa = mlx_openai.chat.completions.create(
            model=mlx_model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": str(CHAMELEON)}},
                {"type": "image_url", "image_url": {"url": str(TOUCAN)}},
                {"type": "text",      "text": VISION_PROMPT},
            ]}],
            max_tokens=256, temperature=0.5,
        )
        results["mlx/openai"] = fmt_wall(t0, resp_oa.usage.completion_tokens)

        t0 = time.time()
        resp_an = mlx_anthropic.messages.create(
            model=mlx_model,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mt_c, "data": b64_c}},
                {"type": "image", "source": {"type": "base64", "media_type": mt_t, "data": b64_t}},
                {"type": "text",  "text": VISION_PROMPT},
            ]}],
            max_tokens=256, temperature=0.5,
        )
        results["mlx/anthropic"] = fmt_wall(t0, resp_an.usage.output_tokens)

    if server in ("ollama", "both"):
        resp_std = std_ollama.chat(
            model=std_model, stream=False,
            messages=[{"role": "user", "content": VISION_PROMPT, "images": [b64_c, b64_t]}],
            options=opts,
        )
        results["std/ollama"] = fmt_ollama(resp_std)

    return results


# =============================================================================
# Speed benchmarks — tool_call
# =============================================================================


def tool_call_speed(mlx_model: str, std_model: str, server: str = "both") -> dict[str, str]:
    msgs = [{"role": "user", "content": TOOL_PROMPT}]
    opts = {"temperature": 0.0, "num_predict": 256}

    results: dict[str, str] = {}

    if server in ("mlx", "both"):
        resp = mlx_ollama.chat(
            model=mlx_model, stream=False, messages=msgs,
            tools=[WEATHER_TOOL_OPENAI], options=opts,
        )
        results["mlx/ollama"] = fmt_ollama(resp)

        t0 = time.time()
        resp_oa = mlx_openai.chat.completions.create(
            model=mlx_model, messages=msgs,
            tools=[WEATHER_TOOL_OPENAI], max_tokens=256, temperature=0.0,
        )
        out_tok = resp_oa.usage.completion_tokens if resp_oa.usage else 0
        results["mlx/openai"] = fmt_wall(t0, out_tok)

        t0 = time.time()
        resp_an = mlx_anthropic.messages.create(
            model=mlx_model, messages=msgs,
            tools=[WEATHER_TOOL_ANTHROPIC], max_tokens=256, temperature=0.0,
        )
        results["mlx/anthropic"] = fmt_wall(t0, resp_an.usage.output_tokens)

    if server in ("ollama", "both"):
        resp_std = std_ollama.chat(
            model=std_model, stream=False, messages=msgs,
            tools=[WEATHER_TOOL_OPENAI], options=opts,
        )
        results["std/ollama"] = fmt_ollama(resp_std)

    return results


# =============================================================================
# Runner
# =============================================================================

BENCHMARKS = [
    ("text",      text_speed),
    ("vision",    vision_speed),
    ("tool_call", tool_call_speed),
]

MLX_ENDPOINTS    = ["mlx/ollama", "mlx/openai", "mlx/anthropic"]
OLLAMA_ENDPOINTS = ["std/ollama"]
ALL_ENDPOINTS    = MLX_ENDPOINTS + OLLAMA_ENDPOINTS


def run_benchmark(label: str, fn, mlx_model: str, std_model: str, server: str):
    mlx_short = mlx_model.split("/")[-1]
    std_short  = std_model
    print(f"\n{'=' * 64}")
    if server == "mlx":
        print(f"{label.upper()}  —  {mlx_short}  (MLX-NIM)")
    elif server == "ollama":
        print(f"{label.upper()}  —  {std_short}  (Standard Ollama)")
    else:
        print(f"{label.upper()}  —  {mlx_short}  vs  {std_short}")
    print("=" * 64)
    try:
        results = fn(mlx_model, std_model, server)
        endpoints = (
            MLX_ENDPOINTS if server == "mlx"
            else OLLAMA_ENDPOINTS if server == "ollama"
            else ALL_ENDPOINTS
        )
        for ep in endpoints:
            stat = results.get(ep, "—")
            print(f"  {ep:<16}  {stat}")
    except Exception:
        print("  FAIL")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX-NIM vs Ollama speed benchmarks")
    parser.add_argument(
        "--server",
        choices=["mlx", "ollama", "both"],
        default="both",
        help="Which server(s) to benchmark (default: both)",
    )
    args = parser.parse_args()

    for mlx_model, std_model in MODEL_PAIRS:
        for label, fn in BENCHMARKS:
            run_benchmark(label, fn, mlx_model, std_model, args.server)
    print()
