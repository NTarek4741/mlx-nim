import asyncio
import base64
import json
import re
import time
import uuid
from typing import AsyncGenerator
from urllib.request import urlopen

from api.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatRequest,
    Choice,
    FunctionDefinition,
    GenerationOptions,
    ImageContent,
    OllamaMessage,
    OllamaToolCall,
    OllamaToolCallFunction,
    OpenAIAssistantMessage,
    OpenAIMessage,
    OpenAIResponseMessage,
    OpenAISystemMessage,
    OpenAIToolCall,
    OpenAIToolMessage,
    OpenAIUserMessage,
    TextContent,
    Tool,
    Usage,
)
from api.api_utils import GenerationStatsCollector


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
