# api_tests/api_test.py
#
# Prerequisites:
#   pip install anthropic openai
#
# Run with the server already started:
#   python -m api_tests.api_test
#
# Both the Anthropic and OpenAI SDKs are pointed at the local mlx-nim server.
# api_key is required by the SDKs but not validated by the server.

import base64
import json
import traceback
from pathlib import Path

import anthropic
import openai

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
TOUCAN = DEMO_DATA / "toucan.jpeg"

MEDIA_TYPES = {
    ".webp": "image/webp",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}

# =============================================================================
# Clients
# =============================================================================

anthropic_client = anthropic.Anthropic(
    base_url=BASE_URL,
    api_key="x",
)

openai_client = openai.OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="x",
)

# =============================================================================
# Helpers
# =============================================================================


def load_image_b64(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    media_type = MEDIA_TYPES[path.suffix.lower()]
    return b64, media_type


def run_test(label: str, fn, *args):
    """Run a test function, printing PASS/FAIL and a response summary."""
    print(f"\n  [{label}]")
    try:
        result = fn(*args)
        print(f"  PASS — {result}")
    except Exception:
        print(f"  FAIL")
        traceback.print_exc()


# =============================================================================
# Test 1 — Text + system prompt + inference params
# =============================================================================

SYSTEM_PROMPT = (
    "You are a concise creative writing assistant. "
    "Respond in no more than two sentences."
)

TEXT_PROMPT = "Write a one-sentence description of a distant star."


def test_text_anthropic(model: str) -> str:
    response = anthropic_client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": TEXT_PROMPT}],
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        stop_sequences=["END"],
    )
    text = response.content[0].text
    assert text, "Expected non-empty text response"
    return text[:120]


def test_text_openai(model: str) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEXT_PROMPT},
        ],
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
        stop=["END"],
        seed=42,
    )
    text = response.choices[0].message.content
    assert text, "Expected non-empty text response"
    return text[:120]


# =============================================================================
# Test 2 — Vision: compare two images
# =============================================================================

VISION_PROMPT = (
    "You are looking at two images. Briefly describe what is in each image "
    "and name one key visual difference between them."
)


def test_vision_anthropic(model: str) -> str:
    chameleon_b64, chameleon_type = load_image_b64(CHAMELEON)
    toucan_b64, toucan_type = load_image_b64(TOUCAN)

    response = anthropic_client.messages.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": chameleon_type,
                            "data": chameleon_b64,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": toucan_type,
                            "data": toucan_b64,
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        max_tokens=256,
        temperature=0.5,
    )
    text = response.content[0].text
    assert text, "Expected non-empty vision response"
    return text[:120]


def test_vision_openai(model: str) -> str:
    # The server's openai_to_chat_convert reads image_url.url as a local file path
    # and converts it to base64 internally — send the absolute path directly.
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": str(CHAMELEON)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": str(TOUCAN)},
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        max_tokens=256,
        temperature=0.5,
    )
    text = response.choices[0].message.content
    assert text, "Expected non-empty vision response"
    return text[:120]


# =============================================================================
# Test 3 — Tool call
# =============================================================================

WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g. Paris",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
            },
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
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. Paris",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}

TOOL_PROMPT = "What is the current weather in Paris? Use the available tool."


def test_tool_call_anthropic(model: str) -> str:
    response = anthropic_client.messages.create(
        model=model,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_ANTHROPIC],
        max_tokens=256,
        temperature=0.0,
    )
    tool_blocks = [b for b in response.content if b.type == "tool_use"]
    assert tool_blocks, f"Expected tool_use block, got: {response.content}"
    tc = tool_blocks[0]
    return f"tool={tc.name} input={tc.input}"


def test_tool_call_openai(model: str) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=[WEATHER_TOOL_OPENAI],
        max_tokens=256,
        temperature=0.0,
    )
    msg = response.choices[0].message
    assert msg.tool_calls, f"Expected tool_calls, got finish_reason={response.choices[0].finish_reason}"
    tc = msg.tool_calls[0]
    return f"tool={tc.function.name} args={tc.function.arguments[:80]}"


# =============================================================================
# Test 4 — JSON schema
# =============================================================================

RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "ingredients": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "prep_time_minutes": {"type": "integer"},
    },
    "required": ["name", "ingredients", "steps", "prep_time_minutes"],
}

JSON_PROMPT = "Give me a simple recipe for scrambled eggs."


def test_json_schema_anthropic(model: str) -> str:
    # json_schema is a custom mlx-nim extension on MessagesParams passed via extra_body
    response = anthropic_client.messages.create(
        model=model,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        max_tokens=512,
        temperature=0.3,
        extra_body={"json_schema": json.dumps(RECIPE_SCHEMA)},
    )
    raw = response.content[0].text
    parsed = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    assert "name" in parsed and "ingredients" in parsed, f"Missing required keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}"


def test_json_schema_openai(model: str) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        max_tokens=512,
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "recipe",
                "schema": RECIPE_SCHEMA,
                "strict": True,
            },
        },
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    assert "name" in parsed and "ingredients" in parsed, f"Missing required keys: {parsed.keys()}"
    return f"name={parsed['name']!r} ingredients={len(parsed['ingredients'])}"


# =============================================================================
# Runner
# =============================================================================

TESTS = [
    ("text/anthropic",       test_text_anthropic),
    ("text/openai",          test_text_openai),
    ("vision/anthropic",     test_vision_anthropic),
    ("vision/openai",        test_vision_openai),
    ("tool_call/anthropic",  test_tool_call_anthropic),
    ("tool_call/openai",     test_tool_call_openai),
    ("json_schema/anthropic", test_json_schema_anthropic),
    ("json_schema/openai",   test_json_schema_openai),
]

if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'=' * 64}")
        print(f"Model: {model}")
        print("=" * 64)
        for label, fn in TESTS:
            run_test(label, fn, model)
