import asyncio
import json
import re
import uuid
from typing import AsyncGenerator

from api.api_models import (
    AnthropicMessageResponse,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    ChatRequest,
    FunctionDefinition,
    FunctionParameters,
    GenerationOptions,
    ImageBlockParam,
    MessageParam,
    MessagesParams,
    OllamaMessage,
    OllamaToolCall,
    OllamaToolCallFunction,
    TextBlockParam,
    Tool,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from api.api_utils import GenerationStatsCollector
from api.openai_api_utils import image_url_to_base64


# =============================================================================
# Anthropic Utility Functions
# =============================================================================

TOOL_CALL_PREFIXES = ["<tool_call>", "[TOOL_CALLS]"]


def parse_tool_calls(text: str) -> tuple[list[AnthropicToolUseBlock] | None, str]:
    """
    Parse tool calls from generated text, returning native Anthropic tool_use blocks.
    Mirrors parse_tool() from openai_api_utils but returns (tool_calls, remaining_content)
    instead of stream chunks, so callers can build Anthropic tool_use content blocks.
    """
    if "<tool_call>" in text:
        function_match = re.search(r'<function=([^>]+)>', text)
        function_name = function_match.group(1) if function_match else ""
        params = {}
        for match in re.finditer(r'<parameter=([^>]+)>([^<]+)</parameter>', text):
            params[match.group(1)] = match.group(2).strip()
        remaining = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
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
        remaining = re.sub(r'\[TOOL_CALLS\].*', '', text, flags=re.DOTALL).strip()
    else:
        return None, text

    if not function_name:
        return None, text

    tool_call = AnthropicToolUseBlock(
        type="tool_use",
        id=f"toolu_{uuid.uuid4().hex[:24]}",
        name=function_name,
        input=params,
    )
    return [tool_call], remaining


def system_blocks_to_content(system: str | list[TextBlockParam] | None) -> str | list[dict] | None:
    """
    Convert Anthropic system prompt to string or ordered list of content blocks.
    Stable (cache_control) blocks come first to maximize KV cache hits.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system
    stable = [{"type": "text", "text": b.text} for b in system if b.cache_control]
    dynamic = [{"type": "text", "text": b.text} for b in system if not b.cache_control]
    blocks = stable + dynamic
    if len(blocks) == 1:
        return blocks[0]["text"]
    return blocks


def anthropic_to_chat_convert(params: MessagesParams) -> ChatRequest:
    """
    Convert Anthropic MessagesParams to internal ChatRequest format.
    Mirrors the pattern of openai_to_chat_convert().
    """
    chat_messages = []

    # Handle system prompt (Anthropic has it as a separate field)
    system_content = system_blocks_to_content(params.system)
    if system_content:
        chat_messages.append(
            OllamaMessage(
                role="system",
                content=system_content,
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
    tool_calls: list[AnthropicToolUseBlock] | None = None,
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
        content.extend(tool_calls)
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
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

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
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tc.id, 'name': tc.name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tc.input)}})}\n\n"
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
