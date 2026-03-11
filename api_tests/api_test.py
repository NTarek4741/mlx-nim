# api_tests/api_test.py
#
# Prerequisites:
#   pip install anthropic openai ollama requests
#
# Run with the server already started:
#   python -m api_tests.api_test
#
# Tests are grouped by capability (text, vision, tool_call, json_schema, logprobs).
# Each capability runs non-streaming and streaming variants for all three endpoints.

import base64
import json
import time
import traceback
from pathlib import Path

import anthropic
import ollama
import openai
import requests

# =============================================================================
# Config
# =============================================================================

BASE_URL = "http://localhost:1234"

MODELS = [
    "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
    "qwen/Qwen3.5-4B-MLX-4bit",
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

anthropic_client = anthropic.Anthropic(base_url=BASE_URL, api_key="x")
openai_client    = openai.OpenAI(base_url=f"{BASE_URL}/v1", api_key="x")
ollama_client    = ollama.Client(host=BASE_URL)

# =============================================================================
# Helpers
# =============================================================================


def load_image_b64(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    data      = path.read_bytes()
    b64       = base64.standard_b64encode(data).decode()
    media_type = MEDIA_TYPES[path.suffix.lower()]
    return b64, media_type


def run_test(label: str, fn, *args):
    """Run a test function, printing PASS/FAIL and a response summary."""
    print(f"\n  [{label}]")
    try:
        result = fn(*args)
        print(f"  PASS — {result}")
    except Exception:
        print("  FAIL")
        traceback.print_exc()


def ollama_speed_str(resp) -> str:
    """Format gen-stats from an Ollama response (streaming final chunk or non-streaming)."""
    parts = []
    if getattr(resp, "load_duration", None):
        parts.append(f"load={resp.load_duration / 1e9:.2f}s")
    if getattr(resp, "prompt_eval_count", None) and getattr(resp, "prompt_eval_duration", None):
        pp = resp.prompt_eval_count / (resp.prompt_eval_duration / 1e9)
        parts.append(f"pp={pp:.0f}t/s")
    if getattr(resp, "eval_count", None) and getattr(resp, "eval_duration", None):
        gen = resp.eval_count / (resp.eval_duration / 1e9)
        parts.append(f"gen={gen:.0f}t/s")
    return "  ".join(parts)


def sdk_speed_str(start: float, in_tokens: int, out_tokens: int) -> str:
    """Wall-clock-based gen speed for Anthropic/OpenAI (server timing not exposed)."""
    elapsed = time.time() - start
    if elapsed > 0 and out_tokens:
        return f"gen={out_tokens / elapsed:.0f}t/s  ({in_tokens}→{out_tokens} tok)"
    return f"elapsed={elapsed:.1f}s"


# =============================================================================
# Shared prompts, tools, and schemas
# =============================================================================

SYSTEM_PROMPT = (
    "You are a concise creative writing assistant. "
    "Respond in no more than two sentences."
)
TEXT_PROMPT    = "Write a one-sentence description of a distant star."
VISION_PROMPT  = (
    "You are looking at two images. Briefly describe what is in each image "
    "and name one key visual difference between them."
)
TOOL_PROMPT    = "What is the current weather in Paris? Use the available tool."
JSON_PROMPT    = "Give me a simple recipe for scrambled eggs."
LOGPROBS_PROMPT = "Name one planet in our solar system."

WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city name, e.g. Paris"},
            "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
        },
        "required": ["location"],
    },
}

WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name, e.g. Paris"},
                "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["location"],
        },
    },
}

RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "name":              {"type": "string"},
        "ingredients":       {"type": "array", "items": {"type": "string"}},
        "steps":             {"type": "array", "items": {"type": "string"}},
        "prep_time_minutes": {"type": "integer"},
    },
    "required": ["name", "ingredients", "steps", "prep_time_minutes"],
}

# =============================================================================
# Text — system prompt + inference params
# =============================================================================

_TEXT_OPTIONS        = {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "num_predict": 128, "stop": ["END"]}
_TEXT_STREAM_OPTIONS = {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "num_predict": 512, "stop": ["END"]}
_TEXT_MSGS_OLLAMA = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": TEXT_PROMPT},
]


def test_text_ollama(model: str) -> str:
    resp = ollama_client.chat(model=model, messages=_TEXT_MSGS_OLLAMA,
                              stream=False, options=_TEXT_OPTIONS)
    assert resp.message.content, "Expected non-empty text"
    return f"{resp.message.content[:100]}  |  {ollama_speed_str(resp)}"


def test_text_ollama_stream(model: str) -> str:
    content, final = "", None
    for chunk in ollama_client.chat(model=model, messages=_TEXT_MSGS_OLLAMA,
                                    stream=True, options=_TEXT_STREAM_OPTIONS):
        content += chunk.message.content or ""
        if chunk.done:
            final = chunk
    assert content, "Expected non-empty streamed text"
    return f"{content[:100]}  |  {ollama_speed_str(final)}"


def test_text_openai(model: str) -> str:
    t0 = time.time()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": TEXT_PROMPT}],
        max_tokens=128, temperature=0.7, top_p=0.9, stop=["END"], seed=42,
    )
    text = resp.choices[0].message.content
    assert text, "Expected non-empty text"
    return f"{text[:100]}  |  {sdk_speed_str(t0, resp.usage.prompt_tokens, resp.usage.completion_tokens)}"


def test_text_openai_stream(model: str) -> str:
    t0, content, usage = time.time(), "", None
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": TEXT_PROMPT}],
        max_tokens=512, temperature=0.7, top_p=0.9, stop=["END"], seed=42,
        stream=True, stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            elif getattr(delta, "reasoning_content", None):
                content += delta.reasoning_content  # fallback for thinking models
        if chunk.usage:
            usage = chunk.usage
    assert content, "Expected non-empty streamed text"
    speed = sdk_speed_str(t0, usage.prompt_tokens, usage.completion_tokens) if usage else f"elapsed={time.time()-t0:.1f}s"
    return f"{content[:100]}  |  {speed}"


def test_text_anthropic(model: str) -> str:
    t0 = time.time()
    resp = anthropic_client.messages.create(
        model=model, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": TEXT_PROMPT}],
        max_tokens=128, temperature=0.7, top_p=0.9, top_k=50, stop_sequences=["END"],
    )
    text = resp.content[0].text
    assert text, "Expected non-empty text"
    return f"{text[:100]}  |  {sdk_speed_str(t0, resp.usage.input_tokens, resp.usage.output_tokens)}"


def test_text_anthropic_stream(model: str) -> str:
    t0 = time.time()
    with anthropic_client.messages.stream(
        model=model, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": TEXT_PROMPT}],
        max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop_sequences=["END"],
    ) as stream:
        text = stream.get_final_text()
        msg  = stream.get_final_message()
    assert text, "Expected non-empty streamed text"
    return f"{text[:100]}  |  {sdk_speed_str(t0, msg.usage.input_tokens, msg.usage.output_tokens)}"


# =============================================================================
# Vision — compare two images
# =============================================================================


def test_vision_ollama(model: str) -> str:
    b64_c, _ = load_image_b64(CHAMELEON)
    b64_t, _ = load_image_b64(TOUCAN)
    resp = ollama_client.chat(
        model=model, stream=False,
        messages=[{"role": "user", "content": VISION_PROMPT, "images": [b64_c, b64_t]}],
        options={"temperature": 0.5, "num_predict": 256},
    )
    assert resp.message.content, "Expected non-empty vision response"
    return f"{resp.message.content[:100]}  |  {ollama_speed_str(resp)}"


def test_vision_ollama_stream(model: str) -> str:
    b64_c, _ = load_image_b64(CHAMELEON)
    b64_t, _ = load_image_b64(TOUCAN)
    content, final = "", None
    for chunk in ollama_client.chat(
        model=model, stream=True,
        messages=[{"role": "user", "content": VISION_PROMPT, "images": [b64_c, b64_t]}],
        options={"temperature": 0.5, "num_predict": 1024},
    ):
        content += chunk.message.content or ""
        if chunk.done:
            final = chunk
    assert content, "Expected non-empty streamed vision response"
    return f"{content[:100]}  |  {ollama_speed_str(final)}"


def test_vision_openai(model: str) -> str:
    t0 = time.time()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": str(CHAMELEON)}},
            {"type": "image_url", "image_url": {"url": str(TOUCAN)}},
            {"type": "text",      "text": VISION_PROMPT},
        ]}],
        max_tokens=256, temperature=0.5,
    )
    text = resp.choices[0].message.content
    assert text, "Expected non-empty vision response"
    return f"{text[:100]}  |  {sdk_speed_str(t0, resp.usage.prompt_tokens, resp.usage.completion_tokens)}"


def test_vision_openai_stream(model: str) -> str:
    t0, content, usage = time.time(), "", None
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": str(CHAMELEON)}},
            {"type": "image_url", "image_url": {"url": str(TOUCAN)}},
            {"type": "text",      "text": VISION_PROMPT},
        ]}],
        max_tokens=1024, temperature=0.5,
        stream=True, stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            elif getattr(delta, "reasoning_content", None):
                content += delta.reasoning_content  # fallback for thinking models
        if chunk.usage:
            usage = chunk.usage
    assert content, "Expected non-empty streamed vision response"
    speed = sdk_speed_str(t0, usage.prompt_tokens, usage.completion_tokens) if usage else f"elapsed={time.time()-t0:.1f}s"
    return f"{content[:100]}  |  {speed}"


def test_vision_anthropic(model: str) -> str:
    t0 = time.time()
    b64_c, mt_c = load_image_b64(CHAMELEON)
    b64_t, mt_t = load_image_b64(TOUCAN)
    resp = anthropic_client.messages.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt_c, "data": b64_c}},
            {"type": "image", "source": {"type": "base64", "media_type": mt_t, "data": b64_t}},
            {"type": "text",  "text": VISION_PROMPT},
        ]}],
        max_tokens=256, temperature=0.5,
    )
    text = resp.content[0].text
    assert text, "Expected non-empty vision response"
    return f"{text[:100]}  |  {sdk_speed_str(t0, resp.usage.input_tokens, resp.usage.output_tokens)}"


def test_vision_anthropic_stream(model: str) -> str:
    t0 = time.time()
    b64_c, mt_c = load_image_b64(CHAMELEON)
    b64_t, mt_t = load_image_b64(TOUCAN)
    with anthropic_client.messages.stream(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt_c, "data": b64_c}},
            {"type": "image", "source": {"type": "base64", "media_type": mt_t, "data": b64_t}},
            {"type": "text",  "text": VISION_PROMPT},
        ]}],
        max_tokens=1024, temperature=0.5,
    ) as stream:
        text = stream.get_final_text()
        msg  = stream.get_final_message()
    assert text, "Expected non-empty streamed vision response"
    return f"{text[:100]}  |  {sdk_speed_str(t0, msg.usage.input_tokens, msg.usage.output_tokens)}"


# =============================================================================
# Tool call
# =============================================================================


def test_tool_call_ollama(model: str) -> str:
    resp = ollama_client.chat(
        model=model, stream=False,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_OPENAI],
        options={"temperature": 0.0, "num_predict": 256},
    )
    tcs = resp.message.tool_calls
    assert tcs, f"Expected tool_calls, got: {resp.message}"
    tc = tcs[0]
    return f"tool={tc.function.name} args={json.dumps(tc.function.arguments)[:80]}  |  {ollama_speed_str(resp)}"


def test_tool_call_ollama_stream(model: str) -> str:
    final = None
    for chunk in ollama_client.chat(
        model=model, stream=True,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_OPENAI],
        options={"temperature": 0.0, "num_predict": 256},
    ):
        if chunk.done:
            final = chunk
    assert final, "No final chunk received"
    tcs = final.message.tool_calls
    assert tcs, f"Expected tool_calls in final chunk, got: {final.message}"
    tc = tcs[0]
    return f"tool={tc.function.name} args={json.dumps(tc.function.arguments)[:80]}  |  {ollama_speed_str(final)}"


def test_tool_call_openai(model: str) -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_OPENAI], max_tokens=256, temperature=0.0,
    )
    msg = resp.choices[0].message
    assert msg.tool_calls, f"Expected tool_calls, got finish_reason={resp.choices[0].finish_reason}"
    tc = msg.tool_calls[0]
    return f"tool={tc.function.name} args={tc.function.arguments[:80]}"


def test_tool_call_openai_stream(model: str) -> str:
    tool_name, tool_args = "", ""
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_OPENAI], max_tokens=256, temperature=0.0, stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_name += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_args += tc_delta.function.arguments
    assert tool_name, "Expected tool call in stream"
    return f"tool={tool_name} args={tool_args[:80]}"


def test_tool_call_anthropic(model: str) -> str:
    resp = anthropic_client.messages.create(
        model=model, messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_ANTHROPIC], max_tokens=256, temperature=0.0,
    )
    tcs = [b for b in resp.content if b.type == "tool_use"]
    assert tcs, f"Expected tool_use block, got: {resp.content}"
    tc = tcs[0]
    return f"tool={tc.name} input={tc.input}"


def test_tool_call_anthropic_stream(model: str) -> str:
    with anthropic_client.messages.stream(
        model=model, messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_ANTHROPIC], max_tokens=256, temperature=0.0,
    ) as stream:
        msg = stream.get_final_message()
    tcs = [b for b in msg.content if b.type == "tool_use"]
    assert tcs, f"Expected tool_use block in stream, got: {msg.content}"
    tc = tcs[0]
    return f"tool={tc.name} input={tc.input}"


# =============================================================================
# JSON schema
# =============================================================================


def _parse_recipe(raw: str) -> dict:
    return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])


def test_json_schema_ollama(model: str) -> str:
    resp = ollama_client.chat(
        model=model, stream=False,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        format=RECIPE_SCHEMA,
        options={"temperature": 0.3, "num_predict": 512},
    )
    parsed = _parse_recipe(resp.message.content)
    assert "name" in parsed and "ingredients" in parsed, f"Missing keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}  |  {ollama_speed_str(resp)}"


def test_json_schema_ollama_stream(model: str) -> str:
    content, final = "", None
    for chunk in ollama_client.chat(
        model=model, stream=True,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        format=RECIPE_SCHEMA,
        options={"temperature": 0.3, "num_predict": 512},
    ):
        content += chunk.message.content or ""
        if chunk.done:
            final = chunk
    parsed = _parse_recipe(content)
    assert "name" in parsed and "ingredients" in parsed, f"Missing keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}  |  {ollama_speed_str(final)}"


def test_json_schema_openai(model: str) -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        max_tokens=512, temperature=0.3,
        response_format={"type": "json_schema", "json_schema": {"name": "recipe", "schema": RECIPE_SCHEMA, "strict": True}},
    )
    parsed = _parse_recipe(resp.choices[0].message.content)
    assert "name" in parsed and "ingredients" in parsed, f"Missing keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}"


def test_json_schema_openai_stream(model: str) -> str:
    content = ""
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        max_tokens=512, temperature=0.3,
        response_format={"type": "json_schema", "json_schema": {"name": "recipe", "schema": RECIPE_SCHEMA, "strict": True}},
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    parsed = _parse_recipe(content)
    assert "name" in parsed and "ingredients" in parsed, f"Missing keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}"


def test_json_schema_anthropic(model: str) -> str:
    resp = anthropic_client.messages.create(
        model=model, messages=[{"role": "user", "content": JSON_PROMPT}],
        max_tokens=512, temperature=0.3,
        extra_body={"json_schema": json.dumps(RECIPE_SCHEMA)},
    )
    parsed = _parse_recipe(resp.content[0].text)
    assert "name" in parsed and "ingredients" in parsed, f"Missing keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}"


def test_json_schema_anthropic_stream(model: str) -> str:
    with anthropic_client.messages.stream(
        model=model, messages=[{"role": "user", "content": JSON_PROMPT}],
        max_tokens=512, temperature=0.3,
        extra_body={"json_schema": json.dumps(RECIPE_SCHEMA)},
    ) as stream:
        text = stream.get_final_text()
    parsed = _parse_recipe(text)
    assert "name" in parsed and "ingredients" in parsed, f"Missing keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}"


# =============================================================================
# Logprobs  (non-streaming — logprobs are only meaningful on the final response)
# =============================================================================


def test_logprobs_ollama(model: str) -> str:
    # logprobs is a custom mlx-nim extension; send via raw HTTP since the
    # Ollama SDK does not expose it as a named parameter.
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": LOGPROBS_PROMPT}],
            "options": {"temperature": 0.0, "num_predict": 32},
            "stream": False,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    lp = data.get("logprobs")
    assert lp, "Expected non-empty logprobs list"
    entry = lp[0]
    assert {"token", "logprob", "bytes"} <= entry.keys(), f"Missing fields: {entry.keys()}"
    assert entry.get("top_logprobs"), "Expected top_logprobs in first entry"
    return f"tokens={len(lp)} top={len(entry['top_logprobs'])} first={entry['token']!r}({entry['logprob']:.3f})"


def test_logprobs_openai(model: str) -> str:
    resp = openai_client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": LOGPROBS_PROMPT}],
        max_tokens=32, temperature=0.0, logprobs=True, top_logprobs=3,
    )
    lp = resp.choices[0].logprobs and resp.choices[0].logprobs.content
    assert lp, "Expected logprobs.content in response"
    entry = lp[0]
    return f"tokens={len(lp)} first={entry.token!r}({entry.logprob:.3f})"


def test_logprobs_anthropic(model: str) -> str:
    # top_logprobs is a native MessagesParams field in mlx-nim; pass via extra_body
    # so the standard SDK doesn't reject it as an unknown param.
    resp = anthropic_client.messages.create(
        model=model, messages=[{"role": "user", "content": LOGPROBS_PROMPT}],
        max_tokens=32, temperature=0.0,
        extra_body={"top_logprobs": 3},
    )
    text = resp.content[0].text
    assert text, "Expected non-empty response"
    return f"text={text[:80]!r}"


# =============================================================================
# Runner
# =============================================================================

TESTS = [
    # ── Text: system prompt + inference params ────────────────────────────────
    ("--- text ---",                 None),
    ("text/ollama",                  test_text_ollama),
    ("text/ollama/stream",           test_text_ollama_stream),
    ("text/openai",                  test_text_openai),
    ("text/openai/stream",           test_text_openai_stream),
    ("text/anthropic",               test_text_anthropic),
    ("text/anthropic/stream",        test_text_anthropic_stream),
    # ── Vision: compare two images ────────────────────────────────────────────
    ("--- vision ---",               None),
    ("vision/ollama",                test_vision_ollama),
    ("vision/ollama/stream",         test_vision_ollama_stream),
    ("vision/openai",                test_vision_openai),
    ("vision/openai/stream",         test_vision_openai_stream),
    ("vision/anthropic",             test_vision_anthropic),
    ("vision/anthropic/stream",      test_vision_anthropic_stream),
    # ── Tool call ─────────────────────────────────────────────────────────────
    ("--- tool_call ---",            None),
    ("tool_call/ollama",             test_tool_call_ollama),
    ("tool_call/ollama/stream",      test_tool_call_ollama_stream),
    ("tool_call/openai",             test_tool_call_openai),
    ("tool_call/openai/stream",      test_tool_call_openai_stream),
    ("tool_call/anthropic",          test_tool_call_anthropic),
    ("tool_call/anthropic/stream",   test_tool_call_anthropic_stream),
    # ── JSON schema ───────────────────────────────────────────────────────────
    ("--- json_schema ---",          None),
    ("json_schema/ollama",           test_json_schema_ollama),
    ("json_schema/ollama/stream",    test_json_schema_ollama_stream),
    ("json_schema/openai",           test_json_schema_openai),
    ("json_schema/openai/stream",    test_json_schema_openai_stream),
    ("json_schema/anthropic",        test_json_schema_anthropic),
    ("json_schema/anthropic/stream", test_json_schema_anthropic_stream),
    # ── Logprobs ──────────────────────────────────────────────────────────────
    ("--- logprobs ---",             None),
    ("logprobs/ollama",              test_logprobs_ollama),
    ("logprobs/openai",              test_logprobs_openai),
    ("logprobs/anthropic",           test_logprobs_anthropic),
]


if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'=' * 64}")
        print(f"Model: {model}")
        print("=" * 64)
        for label, fn in TESTS:
            if fn is None:
                print(f"\n  {label}")
            else:
                run_test(label, fn, model)
