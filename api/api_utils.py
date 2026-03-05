import asyncio
import base64
import gc
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from urllib.request import urlopen

import mlx as mx
from transformers import AutoProcessor, AutoTokenizer

from api.api_models import (
    AnthropicMessageResponse,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicTool,
    AnthropicToolUseBlock,
    AnthropicUsage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    DeltaMessage,
    FunctionDefinition,
    FunctionParameters,
    GenerateResponse,
    GenerationOptions,
    ImageBlockParam,
    ImageContent,
    LogprobEntry,
    # Anthropic models
    MessageParam,
    MessagesParams,
    OllamaMessage,
    OllamaResponseMessage,
    OllamaToolCall,
    OllamaToolCallFunction,
    OpenAIAssistantMessage,
    OpenAIFunctionCall,
    OpenAIMessage,
    OpenAIResponseMessage,
    OpenAISystemMessage,
    OpenAIToolCall,
    OpenAIToolMessage,
    OpenAIUserMessage,
    TextBlockParam,
    TextContent,
    Tool,
    ToolResultBlockParam,
    ToolUseBlockParam,
    TopLogprobEntry,
    Usage,
)
from mlx_engine.generate import create_generator, load_draft_model, load_model, tokenize
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.utils.token import Token

model_cache = {"load_params": None, "model_kit": None}


def load_and_cache_model(
    model: str,
    num_ctx: int,
    kv_bits: int,
    kv_group_size: int,
    quantized_kv_start: int,
    draft_model: str | None,
):
    """
    Load and cache a model. Returns a tuple of (model_kit, load_duration_ns).
    load_duration_ns is the time spent loading the model in nanoseconds, or 0 if cached.
    """
    # Logic for Loading and Model Caching
    global model_cache

    load_params = {
        "model_path": f"models/{model}",
        "max_kv_size": num_ctx,
        "kv_bits": kv_bits,
        "kv_group_size": kv_group_size,
        "quantized_kv_start": quantized_kv_start,
        "draft_model": f"models/{draft_model}" if draft_model else None,
    }
    # Load Model from cache or clear cache if new load params requested
    if model_cache.get("load_params") == load_params:
        print("Model already loaded ✓", end="\n", flush=True)
        return model_cache.get("model_kit"), 0  # No load time for cached model
    else:
        if model_cache.get("model_kit") is not None:
            del model_cache["model_kit"]
            del model_cache["load_params"]
            mx.core.clear_cache()
            gc.collect()
            print(
                "New Model Requetsted, Previous Model cleared from Cache ✓",
                end="\n",
                flush=True,
            )

    load_start_time = time.time()
    print("Loading model...", end="\n", flush=True)
    model_kit = load_model(
        load_params["model_path"],
        max_kv_size=load_params["max_kv_size"],
        max_seq_nums=1,
        trust_remote_code=False,
        kv_bits=load_params["kv_bits"],
        kv_group_size=load_params["kv_group_size"],
        quantized_kv_start=load_params["quantized_kv_start"],
    )
    print("\rModel load complete ✓", end="\n", flush=True)

    # Load draft model if specified
    if load_params["draft_model"]:
        print(
            f"Loading draft model: {load_params['draft_model']}...",
            end="\n",
            flush=True,
        )
        load_draft_model(model_kit, load_params["draft_model"])
        print("Draft model loaded ✓", end="\n", flush=True)

    load_end_time = time.time()
    load_duration_ns = int((load_end_time - load_start_time) * 1e9)

    # Update cache
    model_cache["model_kit"] = model_kit
    model_cache["load_params"] = load_params

    print("✅ Model ready for inference!", end="\n", flush=True)
    return model_kit, load_duration_ns


class GenerationStatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0
        self.num_accepted_draft_tokens: int | None = None
        self.load_duration: int | None = None  # Model load time in nanoseconds

    def add_tokens(self, tokens: list[Token]):
        """Record new tokens and their timing."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

        draft_tokens = sum(1 for token in tokens if token.from_draft)
        if self.num_accepted_draft_tokens is None:
            self.num_accepted_draft_tokens = 0
        self.num_accepted_draft_tokens += draft_tokens

        self.total_tokens += len(tokens)

    def print_stats(self):
        """Print generation statistics."""
        end_time = time.time()
        total_time = end_time - self.start_time

        # Check if first token was generated
        if self.first_token_time is None:
            print("\n\nNo tokens generated")
            return

        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token
        tokens_per_second = (
            self.total_tokens / effective_time if effective_time > 0 else float("inf")
        )
        print("\n\nGeneration stats:")
        print(f" - Tokens per second: {tokens_per_second:.2f}")
        if self.num_accepted_draft_tokens is not None:
            print(
                f" - Number of accepted draft tokens: {self.num_accepted_draft_tokens}"
            )
        print(f" - Time to first token: {time_to_first_token:.2f}s")
        print(f" - Total tokens generated: {self.total_tokens}")
        print(f" - Total time: {total_time:.2f}s")

    def get_tokens_per_second(self):
        """Calculate and return tokens per second."""
        end_time = time.time()
        total_time = end_time - self.start_time

        if self.first_token_time is None or total_time == 0:
            return 0

        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token

        return self.total_tokens / effective_time if effective_time > 0 else 0


def build_logprobs(
    tokens: list[Token], top_logprobs: list | None, top_logprobs_count: int
) -> list[LogprobEntry]:
    """
    Build logprob entries from tokens and their top logprobs.
    Matches Ollama's format with UTF-8 bytes populated.
    """
    entries = []
    if not top_logprobs:
        return entries

    for token, candidates in zip(tokens, top_logprobs):
        top_entries = [
            TopLogprobEntry(
                token=candidates[x].text,
                logprob=candidates[x].logprob,
                bytes=list(candidates[x].text.encode("utf-8")),
            )
            for x in range(top_logprobs_count)
        ]

        entries.append(
            LogprobEntry(
                token=token.text,
                logprob=token.logprob,
                bytes=list(token.text.encode("utf-8")),
                top_logprobs=top_entries,
            )
        )

    return entries


async def chat_stream(
    generator,
    model_name: str,
    stats_collector: GenerationStatsCollector,
    prompt_tokens: list[int],
    include_logprobs: bool,
    top_logprobs: int,
):
    """
    Stream chat results as JSON chunks matching Ollama /api/chat format.
    Uses ChatResponse model for consistent formatting.
    """
    full_text = ""
    thinking = False
    buffering = None  # None = undecided, True = buffer for tool calls, False = stream
    pending_chunks = []
    last_generation_result = None

    try:
        for generation_result in generator:
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                print("Client disconnected, stopping generation")
                return

            stats_collector.add_tokens(generation_result.tokens)
            last_generation_result = generation_result

            if generation_result.text:
                # Partition-based think tag handling
                text = generation_result.text
                while text:
                    if thinking:
                        if "</think>" in text:
                            _, _, after = text.partition("</think>")
                            thinking = False
                            text = after
                        else:
                            text = ""
                    elif "<think>" in text:
                        before, _, after = text.partition("<think>")
                        if before:
                            full_text += before
                            if buffering is None:
                                pending_chunks.append(before)
                                if any(p in full_text for p in TOOL_CALL_PREFIXES):
                                    buffering = True
                                elif len(full_text) >= 15:
                                    buffering = False
                                    for c in pending_chunks:
                                        yield (
                                            ChatResponse(
                                                model=model_name,
                                                created_at=datetime.now(
                                                    timezone.utc
                                                ).isoformat(),
                                                message=OllamaResponseMessage(
                                                    role="assistant", content=c
                                                ),
                                                done=False,
                                            ).model_dump_json()
                                            + "\n"
                                        )
                                    pending_chunks.clear()
                            elif buffering is False:
                                yield (
                                    ChatResponse(
                                        model=model_name,
                                        created_at=datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        message=OllamaResponseMessage(
                                            role="assistant", content=before
                                        ),
                                        done=False,
                                    ).model_dump_json()
                                    + "\n"
                                )
                        thinking = True
                        text = after
                    else:
                        full_text += text
                        if buffering is None:
                            pending_chunks.append(text)
                            if any(p in full_text for p in TOOL_CALL_PREFIXES):
                                buffering = True
                            elif len(full_text) >= 15:
                                buffering = False
                                for c in pending_chunks:
                                    yield (
                                        ChatResponse(
                                            model=model_name,
                                            created_at=datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            message=OllamaResponseMessage(
                                                role="assistant", content=c
                                            ),
                                            done=False,
                                        ).model_dump_json()
                                        + "\n"
                                    )
                                pending_chunks.clear()
                        elif buffering is False:
                            yield (
                                ChatResponse(
                                    model=model_name,
                                    created_at=datetime.now(timezone.utc).isoformat(),
                                    message=OllamaResponseMessage(
                                        role="assistant", content=text
                                    ),
                                    done=False,
                                ).model_dump_json()
                                + "\n"
                            )
                        text = ""
    except asyncio.CancelledError:
        print("Client disconnected, stopping generation")
        return

    # Resolve buffering state if still undecided (very short output)
    if buffering is None:
        buffering = any(full_text.startswith(p) for p in TOOL_CALL_PREFIXES)
        if not buffering:
            for c in pending_chunks:
                yield (
                    ChatResponse(
                        model=model_name,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        message=OllamaResponseMessage(role="assistant", content=c),
                        done=False,
                    ).model_dump_json()
                    + "\n"
                )

    # Determine done_reason and handle tool calls
    done_reason = "stop"
    if last_generation_result and last_generation_result.stop_condition:
        done_reason = last_generation_result.stop_condition.stop_reason

    tool_calls_out = None
    content_out = ""
    if buffering:
        tool_calls, remaining_content = parse_tool_calls(full_text)
        if tool_calls:
            done_reason = "tool_calls"
            content_out = remaining_content or ""
            tool_calls_out = [
                OllamaToolCall(
                    function=OllamaToolCallFunction(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments,
                    )
                )
                for tc in tool_calls
            ]
        else:
            content_out = full_text

    # Emit final done=True chunk with stats
    end_time = time.time()
    total_time = end_time - stats_collector.start_time
    final_response = ChatResponse(
        model=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        message=OllamaResponseMessage(
            role="assistant", content=content_out, tool_calls=tool_calls_out
        ),
        done=True,
        done_reason=done_reason,
        total_duration=int(total_time * 1e9),
        load_duration=stats_collector.load_duration,
        prompt_eval_count=len(prompt_tokens),
        eval_count=stats_collector.total_tokens,
    )
    if stats_collector.first_token_time:
        prompt_time = stats_collector.first_token_time - stats_collector.start_time
        final_response.prompt_eval_duration = int(prompt_time * 1e9)
        final_response.eval_duration = int((total_time - prompt_time) * 1e9)
    if (
        include_logprobs
        and last_generation_result
        and last_generation_result.top_logprobs
    ):
        final_response.logprobs = build_logprobs(
            last_generation_result.tokens,
            last_generation_result.top_logprobs,
            top_logprobs,
        )
    yield final_response.model_dump_json() + "\n"


async def generate_output(
    generator,
    stats_collector: GenerationStatsCollector,
    generate_query,
    prompt_tokens: list[int],
):
    """
    Collect full generation output for non-streaming response.
    """
    result_text = ""
    generation_result = None
    logprobs_list = [] if generate_query.logprobs else None
    generated_token_ids = []

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

        # Collect generated token IDs for context
        generated_token_ids.extend(token.id for token in generation_result.tokens)

        if logprobs_list is not None and generation_result.top_logprobs:
            logprobs_list.extend(
                build_logprobs(generation_result.tokens, generation_result.top_logprobs)
            )

    # Thinking process
    thinking_content = None
    if generate_query.think:
        re_think = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = re_think.search(result_text)
        if match:
            extracted_thoughts = match.group(1).strip()
            # Remove the think block from response
            result_text = re_think.sub("", result_text).strip()
            thinking_content = extracted_thoughts

    # Calculate durations and stats
    total_duration = None
    prompt_eval_duration = None
    eval_duration = None
    eval_count = None

    finish_reason = "length"
    if generation_result and generation_result.stop_condition:
        if generation_result.stop_condition.stop_reason == "stop_string":
            finish_reason = "stop"
        elif generation_result.stop_condition.stop_reason == "end_token":
            finish_reason = "end"

    # Calculate timing in nanoseconds
    end_time = time.time()
    total_time_sec = end_time - stats_collector.start_time
    total_duration = int(total_time_sec * 1e9)

    eval_count = stats_collector.total_tokens

    if stats_collector.first_token_time:
        prompt_eval_time_sec = (
            stats_collector.first_token_time - stats_collector.start_time
        )
        prompt_eval_duration = int(prompt_eval_time_sec * 1e9)
        gen_time_sec = total_time_sec - prompt_eval_time_sec
        eval_duration = int(gen_time_sec * 1e9)

    # Construct structured response
    final_logprobs = (
        logprobs_list if (generate_query.logprobs and logprobs_list) else None
    )

    return GenerateResponse(
        model=generate_query.model,
        created_at=datetime.now(timezone.utc).isoformat(),
        response=result_text,
        thinking=thinking_content,
        done=True,
        done_reason=finish_reason,
        total_duration=total_duration,
        load_duration=stats_collector.load_duration,
        prompt_eval_count=len(prompt_tokens),
        prompt_eval_duration=prompt_eval_duration,
        eval_count=eval_count,
        eval_duration=eval_duration,
        logprobs=final_logprobs,
        context=prompt_tokens + generated_token_ids,
    )


async def chat_render(
    messages: list[OllamaMessage], tools: list[Tool] | None, images: list[str]
):
    tf_tokenizer = model_cache["model_kit"].tokenizer._tokenizer

    # Convert Pydantic models to dicts and collect images
    messages_dicts = []
    for msg in messages:
        if msg.images:
            # Build content with image placeholders first, then text
            content = []
            for i in range(len(msg.images)):
                content.append(
                    {"type": "image", "image": "image" + str(len(images) + i)}
                )
            content.append({"type": "text", "text": msg.content})
            images.extend(msg.images)
            messages_dicts.append({"role": msg.role, "content": content})
        else:
            # No images - content is just a string
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_dicts.append(msg_dict)

    tools_dicts = (
        [tool.model_dump(exclude_none=True) for tool in tools] if tools else None
    )

    prompt = tf_tokenizer.apply_chat_template(
        messages_dicts, tools=tools_dicts, tokenize=False, add_generation_prompt=True
    )
    return tokenize(model_cache["model_kit"], prompt)


def openai_to_chat_convert(req: ChatCompletionRequest):
    # Convert OpenAI messages to Chat messages
    chat_messages = []
    for msg in req.messages:
        content = ""
        images = None

        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            # Handle multimodal content
            image_list = []
            for part in msg.content:
                if part.type == "text":
                    content = part.text
                elif part.type == "image_url":
                    # Read file from path and convert to base64
                    file_path = part.image_url.url
                    with open(file_path, "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode("utf-8")
                    image_list.append(base64_data)
            if image_list:
                images = image_list

        # Convert OpenAI tool_calls to Ollama tool_calls
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [
                OllamaToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=OllamaToolCallFunction(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    ),
                )
                for tc in msg.tool_calls
            ]

        # Preserve tool_call_id for tool role messages
        tool_call_id = getattr(msg, "tool_call_id", None)

        chat_messages.append(
            OllamaMessage(
                role=msg.role,
                content=content,
                images=images,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    # Convert OpenAI tools to Chat tools
    chat_tools = None
    if req.tools:
        chat_tools = []
        for tool in req.tools:
            parameters_dict = None
            if tool.function.parameters:
                parameters_dict = tool.function.parameters.model_dump()
            chat_tools.append(
                Tool(
                    type=tool.type,
                    function=FunctionDefinition(
                        name=tool.function.name,
                        description=tool.function.description,
                        parameters=parameters_dict,
                    ),
                )
            )

    # Convert response_format to format
    format_value = None
    if req.response_format:
        if req.response_format.type == "json_object":
            format_value = "json"
        elif req.response_format.type == "json_schema":
            if (
                req.response_format.json_schema
                and req.response_format.json_schema.schema_
            ):
                format_value = req.response_format.json_schema.schema_

    # Build GenerationOptions
    options = GenerationOptions(
        temperature=req.temperature,
        top_p=req.top_p,
        num_predict=req.max_tokens,
        stop=req.stop,
        seed=req.seed,
    )

    # Create and return ChatRequest
    return ChatRequest(
        model=req.model,
        messages=chat_messages,
        tools=chat_tools,
        format=format_value,
        options=options,
        stream=req.stream if req.stream is not None else True,
    )


# =============================================================================
# OpenAI Utility Functions
# =============================================================================


def image_url_to_base64(url: str) -> str:
    """
    Fetch image from URL and convert to base64.
    If already a data URI, extract the base64 part.
    """
    if url.startswith("data:"):
        # Extract base64 from data URI
        # Format: data:image/jpeg;base64,<data>
        if ";base64," in url:
            return url.split(";base64,")[1]
        return url

    # Fetch from URL
    with urlopen(url) as response:
        image_data = response.read()
    return base64.b64encode(image_data).decode("utf-8")


def convert_openai_messages(
    messages: list[OpenAIMessage],
    tools: list[Tool] | None = None,
) -> tuple[list[dict], list[str], list[dict] | None]:
    """
    Convert OpenAI message format to internal format for chat_template.

    Args:
        messages: List of OpenAI Message objects
        tools: Optional list of Tool definitions

    Returns:
        Tuple of (messages_dicts, images_b64, tools_dicts) where:
        - messages_dicts: List of dicts ready for apply_chat_template()
        - images_b64: List of base64-encoded images extracted from content
        - tools_dicts: List of tool definitions in dict format, or None
    """
    messages_dicts = []
    images_b64 = []

    for msg in messages:
        if isinstance(msg, OpenAISystemMessage) or (
            hasattr(msg, "role") and msg.role == "system"
        ):
            messages_dicts.append({"role": "system", "content": msg.content})

        elif isinstance(msg, OpenAIUserMessage) or (
            hasattr(msg, "role") and msg.role == "user"
        ):
            if isinstance(msg.content, str):
                messages_dicts.append({"role": "user", "content": msg.content})
            else:
                # List of content parts (multimodal)
                content_parts = []
                for part in msg.content:
                    if isinstance(part, TextContent) or (
                        hasattr(part, "type") and part.type == "text"
                    ):
                        content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent) or (
                        hasattr(part, "type") and part.type == "image_url"
                    ):
                        image_idx = len(images_b64)
                        content_parts.append(
                            {"type": "image", "image": f"image{image_idx}"}
                        )
                        # Extract base64 from URL or data URI
                        images_b64.append(image_url_to_base64(part.image_url.url))

                # Simplify if only text
                if len(content_parts) == 1 and content_parts[0].get("type") == "text":
                    messages_dicts.append(
                        {"role": "user", "content": content_parts[0]["text"]}
                    )
                else:
                    messages_dicts.append({"role": "user", "content": content_parts})

        elif isinstance(msg, OpenAIAssistantMessage) or (
            hasattr(msg, "role") and msg.role == "assistant"
        ):
            msg_dict = {"role": "assistant", "content": msg.content or ""}
            # Add tool calls if present
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages_dicts.append(msg_dict)

        elif isinstance(msg, OpenAIToolMessage) or (
            hasattr(msg, "role") and msg.role == "tool"
        ):
            messages_dicts.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )

    # Convert tools to dict format
    tools_dicts = None
    if tools:
        tools_dicts = []
        for tool in tools:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description or "",
                    "parameters": {
                        "type": tool.function.parameters.type
                        if tool.function.parameters
                        else "object",
                        "properties": tool.function.parameters.properties
                        if tool.function.parameters
                        else {},
                        "required": tool.function.parameters.required
                        if tool.function.parameters
                        else [],
                    },
                },
            }
            tools_dicts.append(tool_dict)

    return messages_dicts, images_b64, tools_dicts


def build_openai_response(
    model: str,
    result_text: str,
    finish_reason: str,
    prompt_token_count: int,
    completion_token_count: int,
    tool_calls: list[OpenAIToolCall] | None = None,
    reasoning_content: str | None = None,
) -> dict:
    """
    Build OpenAI ChatCompletionResponse from generation results.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=OpenAIResponseMessage(
                    role="assistant",
                    content=result_text if result_text else None,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            total_tokens=prompt_token_count + completion_token_count,
        ),
    )
    return json.loads(response.model_dump_json(exclude_none=True))

def parse_tool(text: str, model_name: str, chunk_id: str, created: int) -> list[str]:
    if "<tool_call>" in text:
        function_match = re.search(r'<function=([^>]+)>', text)
        function_name = function_match.group(1) if function_match else ""
        params = {}
        for match in re.finditer(r'<parameter=([^>]+)>([^<]+)</parameter>', text):
            params[match.group(1)] = match.group(2).strip()
    elif "[TOOL_CALLS]" in text:
        function_name = ""
        params = {}
        match = re.search(r'\[TOOL_CALLS\](\w+)\[ARGS\](\{.*\})', text, re.DOTALL)
        if match:
            function_name = match.group(1)
            try:
                params = json.loads(match.group(2))
            except json.JSONDecodeError:
                pass

    base = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
    }

    name_chunk = {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": chunk_id, "type": "function", "function": {"name": function_name, "arguments": ""}}]}, "logprobs": None, "finish_reason": None}]}
    args_chunk = {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": json.dumps(params)}}]}, "logprobs": None, "finish_reason": None}]}
    finish_chunk = {**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "tool_calls"}]}

    return [
        f"data: {json.dumps(name_chunk)}\n\n",
        f"data: {json.dumps(args_chunk)}\n\n",
        f"data: {json.dumps(finish_chunk)}\n\n",
    ]



async def openai_stream(
    generator,
    model_name: str,
    stats_collector: GenerationStatsCollector,
    prompt_tokens: list[int],
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    thinking = False
    calling_tool = False
    tool_call = ""
    tool_format = None  # "xml" or "mistral"
    chunk_id = str(uuid.uuid4())
    created = int(time.time())

    def make_chunk(delta: dict) -> str:
        return f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': None}]})}\n\n"

    for generation_result in generator:
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

        stats_collector.add_tokens(generation_result.tokens)

        if generation_result.text:
            text = re.sub(r'<\|im_(start|end)\|>', '', generation_result.text)
            if not text:
                continue

            if "<think>" in text:
                thinking = True
                text = text.replace("<think>", "")
            if "</think>" in text:
                thinking = False
                text = text.replace("</think>", "")

            if "<tool_call>" in text or "[TOOL_CALLS]" in text:
                calling_tool = True
            if "</tool_call>" in text:
                tool_call += text
                calling_tool = False
                for chunk in parse_tool(tool_call, model_name, chunk_id, created):
                    yield chunk
                return
            if calling_tool:
                tool_call += text
                continue

            if thinking:
                yield make_chunk({"reasoning_content": text})
            else:
                yield make_chunk({"content": text})

    if tool_call and "[TOOL_CALLS]" in tool_call:
        print(tool_call)
        for chunk in parse_tool(tool_call, model_name, chunk_id, created):
            yield chunk
        return


    stop_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"






































































































async def generate_openai_output(
    generator,
    stats_collector: GenerationStatsCollector,
    params: ChatCompletionRequest,
    prompt_token_count: int,
) -> ChatCompletionResponse:
    """
    Collect full generation output and return OpenAI response.
    Extracts thinking blocks as reasoning_content.
    """
    result_text = ""
    generation_result = None

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

    # Extract thinking blocks as reasoning_content
    reasoning_content = None
    think_matches = re.findall(r"<think>(.*?)</think>", result_text, flags=re.DOTALL)
    if think_matches:
        reasoning_content = "\n".join(m.strip() for m in think_matches)
        result_text = re.sub(
            r"<think>.*?</think>", "", result_text, flags=re.DOTALL
        ).strip()

    # Determine finish reason
    finish_reason = "length"
    if generation_result and generation_result.stop_condition:
        stop_reason = generation_result.stop_condition.stop_reason
        if stop_reason in ("stop_string", "eos_token"):
            finish_reason = "stop"

    # Parse tool calls from generated text
    tool_calls, remaining_content = parse_tool_calls(result_text)
    if tool_calls:
        finish_reason = "tool_calls"
        reasoning_content = None  # drop thinking content when returning tool calls

    return build_openai_response(
        model=params.model,
        result_text=remaining_content,
        finish_reason=finish_reason,
        prompt_token_count=prompt_token_count,
        completion_token_count=stats_collector.total_tokens,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )


# =============================================================================
# Anthropic Utility Functions
# =============================================================================


def normalize_system_prompt(system: str | list[TextBlockParam] | None) -> str | None:
    """
    Convert Anthropic system prompt format to plain string.

    Args:
        system: Can be None, str, or list of TextBlockParam

    Returns:
        Plain string system prompt or None
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system
    # List of TextBlockParam - concatenate all text
    return "\n".join(block.text for block in system)


def anthropic_to_chat_convert(params: MessagesParams) -> ChatRequest:
    """
    Convert Anthropic MessagesParams to internal ChatRequest format.
    Mirrors the pattern of openai_to_chat_convert().
    """
    chat_messages = []

    # Handle system prompt (Anthropic has it as a separate field)
    system_str = normalize_system_prompt(params.system)
    if system_str:
        chat_messages.append(
            OllamaMessage(
                role="system",
                content=system_str,
            )
        )

    # Convert each Anthropic message
    for msg in params.messages:
        if isinstance(msg.content, str):
            chat_messages.append(
                OllamaMessage(
                    role=msg.role,
                    content=msg.content,
                )
            )
        else:
            # Process content blocks
            text_parts = []
            image_list = []
            tool_call_list = []

            for block in msg.content:
                if isinstance(block, TextBlockParam) or (
                    hasattr(block, "type") and block.type == "text"
                ):
                    text_parts.append(block.text)
                elif isinstance(block, ImageBlockParam) or (
                    hasattr(block, "type") and block.type == "image"
                ):
                    if block.source.type == "base64":
                        image_list.append(block.source.data)
                    elif block.source.type == "url":
                        image_list.append(image_url_to_base64(block.source.url))
                elif isinstance(block, ToolUseBlockParam) or (
                    hasattr(block, "type") and block.type == "tool_use"
                ):
                    tool_call_list.append(
                        OllamaToolCall(
                            id=block.id,
                            type="function",
                            function=OllamaToolCallFunction(
                                name=block.name,
                                arguments=block.input
                                if isinstance(block.input, dict)
                                else json.loads(block.input),
                            ),
                        )
                    )
                elif isinstance(block, ToolResultBlockParam) or (
                    hasattr(block, "type") and block.type == "tool_result"
                ):
                    # Tool results become separate "tool" role messages
                    tool_content = block.content
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            b.text if hasattr(b, "text") else str(b)
                            for b in tool_content
                        )
                    chat_messages.append(
                        OllamaMessage(
                            role="tool",
                            content=tool_content or "",
                            tool_call_id=block.tool_use_id,
                        )
                    )

            content = "\n".join(text_parts) if text_parts else ""
            images = image_list if image_list else None
            tool_calls = tool_call_list if tool_call_list else None

            # Add the main message if there's content or tool calls
            if content or tool_calls:
                chat_messages.append(
                    OllamaMessage(
                        role=msg.role,
                        content=content,
                        images=images,
                        tool_calls=tool_calls,
                    )
                )
            elif images:
                # Image-only message
                chat_messages.append(
                    OllamaMessage(
                        role=msg.role,
                        content="",
                        images=images,
                    )
                )

    # Convert Anthropic tools to Chat tools (OpenAI format)
    chat_tools = None
    if params.tools:
        chat_tools = []
        for tool in params.tools:
            chat_tools.append(
                Tool(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=FunctionParameters(
                            type=tool.input_schema.type,
                            properties=tool.input_schema.properties,
                            required=tool.input_schema.required,
                        )
                        if tool.input_schema
                        else None,
                    ),
                )
            )

    # Build GenerationOptions from Anthropic params
    options = GenerationOptions(
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        num_predict=params.max_tokens,
        stop=params.stop_sequences,
        num_ctx=params.max_kv_size,
        kv_bits=params.kv_bits,
        kv_group_size=params.kv_group_size,
        quantized_kv_start=params.quantized_kv_start,
        draft_model=params.draft_model,
        num_draft_tokens=params.num_draft_tokens,
    )

    # Handle JSON schema
    format_value = None
    if params.json_schema:
        format_value = (
            json.loads(params.json_schema)
            if isinstance(params.json_schema, str)
            else params.json_schema
        )

    return ChatRequest(
        model=params.model,
        messages=chat_messages,
        tools=chat_tools,
        format=format_value,
        options=options,
        stream=params.stream if params.stream is not None else False,
        logprobs=params.top_logprobs > 0 if params.top_logprobs else False,
        top_logprobs=params.top_logprobs or 0,
    )


def build_anthropic_response(
    model: str,
    result_text: str | None,
    stop_reason: str,
    prompt_token_count: int,
    completion_token_count: int,
    tool_calls: list[OpenAIToolCall] | None = None,
    thinking_text: str | None = None,
) -> dict:
    """
    Build native Anthropic Message response from generation results.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    content: list[
        AnthropicThinkingBlock | AnthropicTextBlock | AnthropicToolUseBlock
    ] = []
    if thinking_text:
        content.append(AnthropicThinkingBlock(type="thinking", thinking=thinking_text))
    if result_text:
        content.append(AnthropicTextBlock(type="text", text=result_text))
    if tool_calls:
        for tc in tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            content.append(
                AnthropicToolUseBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                )
            )
    if not content:
        content.append(AnthropicTextBlock(type="text", text=""))

    response = AnthropicMessageResponse(
        id=message_id,
        type="message",
        role="assistant",
        content=content,
        model=model,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=prompt_token_count,
            output_tokens=completion_token_count,
        ),
    )
    return json.loads(response.model_dump_json(exclude_none=True))


async def anthropic_stream(
    generator,
    model_name: str,
    stats_collector: GenerationStatsCollector,
    prompt_tokens: list[int],
    include_logprobs: bool,
    top_logprobs: int,
) -> AsyncGenerator[str, None]:
    """
    Stream generation results in Anthropic SSE format.
    Emits thinking blocks as {"type": "thinking"} content blocks (like real API),
    then text blocks, then tool_use blocks if detected.

    Event sequence:
    - message_start
    - [if thinking:] content_block_start(thinking) -> thinking_deltas -> content_block_stop
    - content_block_start(text) -> text_deltas -> content_block_stop
    - [if tool calls:] content_block_start(tool_use) -> input_json_delta -> content_block_stop
    - message_delta (stop_reason + usage)
    - message_stop
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # 1. message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": len(prompt_tokens), "output_tokens": 0},
        },
    }
    yield f"event: message_startndata: {json.dumps(message_start)}\n\n"

    # Track state
    block_index = 0  # next content block index to emit
    thinking = False  # inside <think> block
    thinking_started = False  # whether we've emitted a thinking content_block_start
    text_started = False  # whether we've emitted a text content_block_start
    text_block_index = -1  # index of the text block (set when emitted)
    full_text = ""
    buffering = None  # None = undecided, True = buffer for tool calls, False = stream
    pending_chunks = []
    stop_reason = None

    try:
        for generation_result in generator:
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                return

            stats_collector.add_tokens(generation_result.tokens)

            if generation_result.text:
                # Partition-based think tag handling — processes content on both sides of tag boundaries
                text = generation_result.text
                while text:
                    if thinking:
                        if "</think>" in text:
                            before, _, after = text.partition("</think>")
                            if before:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': before}})}\n\n"
                            thinking = False
                            if thinking_started:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                                thinking_started = False
                            text = after
                        else:
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': text}})}\n\n"
                            text = ""
                    elif "<think>" in text:
                        before, _, after = text.partition("<think>")
                        if before:
                            # Normal text before the opening think tag
                            full_text += before
                            if buffering is None:
                                pending_chunks.append(before)
                                if any(p in full_text for p in TOOL_CALL_PREFIXES):
                                    buffering = True
                                elif len(full_text) >= 15:
                                    buffering = False
                                    if not text_started:
                                        text_block_index = block_index
                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        text_started = True
                                        block_index += 1
                                    for chunk_text in pending_chunks:
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': chunk_text}})}\n\n"
                                    pending_chunks.clear()
                            elif buffering is False:
                                if not text_started:
                                    text_block_index = block_index
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                    text_started = True
                                    block_index += 1
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': before}})}\n\n"
                        thinking = True
                        if not thinking_started:
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                            thinking_started = True
                        text = after
                    else:
                        # Pure normal text — no think tags in this chunk
                        full_text += text
                        if buffering is None:
                            pending_chunks.append(text)
                            if any(p in full_text for p in TOOL_CALL_PREFIXES):
                                buffering = True
                            elif len(full_text) >= 15:
                                buffering = False
                                if not text_started:
                                    text_block_index = block_index
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                    text_started = True
                                    block_index += 1
                                for chunk_text in pending_chunks:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': chunk_text}})}\n\n"
                                pending_chunks.clear()
                        elif buffering is False:
                            if not text_started:
                                text_block_index = block_index
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                text_started = True
                                block_index += 1
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                        text = ""
                # buffering is True — just accumulate

            if generation_result.stop_condition:
                sr = generation_result.stop_condition.stop_reason
                if sr in ("stop_string", "eos_token"):
                    stop_reason = "end_turn"
                elif sr == "max_tokens":
                    stop_reason = "max_tokens"
                break
    except asyncio.CancelledError:
        return

    # Close any open thinking block (model stopped mid-think)
    if thinking_started:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
        block_index += 1
        thinking_started = False

    # If we never decided (very short output), resolve now
    if buffering is None:
        buffering = any(full_text.startswith(p) for p in TOOL_CALL_PREFIXES)
        if not buffering:
            if not text_started:
                text_block_index = block_index
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_started = True
                block_index += 1
            for chunk_text in pending_chunks:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': chunk_text}})}\n\n"

    # Handle tool calls from buffered text
    if buffering:
        tool_calls, remaining_content = parse_tool_calls(full_text)
        if tool_calls:
            # Start text block if needed, emit remaining text
            if not text_started:
                text_block_index = block_index
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_started = True
                block_index += 1
            if remaining_content:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': remaining_content}})}\n\n"
            # Close text block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"

            # Emit tool_use blocks
            for tc in tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tc.id, 'name': tc.function.name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(args)}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                block_index += 1

            stop_reason = "tool_use"
        else:
            # Looked like tool call but wasn't — flush as text
            if not text_started:
                text_block_index = block_index
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_started = True
                block_index += 1
            if full_text:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': full_text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
    else:
        # No tool calls — ensure text block is started and close it
        if not text_started:
            text_block_index = block_index
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_started = True
            block_index += 1
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"

    # message_delta with stop_reason and usage
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason or 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': stats_collector.total_tokens}})}\n\n"

    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


async def generate_anthropic_output(
    generator,
    stats_collector: GenerationStatsCollector,
    params: MessagesParams,
    prompt_token_count: int,
) -> dict:
    """
    Collect full generation output and return native Anthropic Message response.
    Extracts thinking blocks as {"type": "thinking"} content blocks.
    Parses tool calls into tool_use content blocks.
    """
    result_text = ""
    generation_result = None

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

    # Extract thinking blocks as structured content
    thinking_text = None
    think_matches = re.findall(r"<think>(.*?)</think>", result_text, flags=re.DOTALL)
    if think_matches:
        thinking_text = "\n".join(m.strip() for m in think_matches)
        result_text = re.sub(
            r"<think>.*?</think>", "", result_text, flags=re.DOTALL
        ).strip()

    # Determine stop reason
    stop_reason = "max_tokens"
    if generation_result and generation_result.stop_condition:
        sr = generation_result.stop_condition.stop_reason
        if sr in ("stop_string", "eos_token"):
            stop_reason = "end_turn"

    # Parse tool calls from generated text
    tool_calls, remaining_content = parse_tool_calls(result_text)
    if tool_calls:
        stop_reason = "tool_use"

    return build_anthropic_response(
        model=params.model,
        result_text=remaining_content,
        stop_reason=stop_reason,
        prompt_token_count=prompt_token_count,
        completion_token_count=stats_collector.total_tokens,
        tool_calls=tool_calls,
        thinking_text=thinking_text,
    )
