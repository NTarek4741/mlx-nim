import asyncio
import json
import re
import time
from datetime import datetime, timezone

from api.api_models import (
    ChatRequest,
    ChatResponse,
    OllamaResponseMessage,
    OllamaToolCall,
    OllamaToolCallFunction,
)
from api.api_utils import GenerationStatsCollector, build_logprobs

# Imported here so chat_stream and generate_chat_output can use them directly.
from api.anthropic_api_utils import TOOL_CALL_PREFIXES, parse_tool_calls


# =============================================================================
# Chat Utility Functions  (Ollama /api/chat format)
# =============================================================================


def build_chat_response(
    model: str,
    text: str,
    tool_calls_out: list[OllamaToolCall] | None,
    logprobs,
    done_reason: str,
    stats_collector: GenerationStatsCollector,
    prompt_tokens: list[int],
) -> ChatResponse:
    """Build a final (done=True) ChatResponse from collected generation data."""
    end_time = time.time()
    total_time = end_time - stats_collector.start_time

    response = ChatResponse(
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(),
        message=OllamaResponseMessage(
            role="assistant",
            content=text,
            tool_calls=tool_calls_out,
        ),
        done=True,
        done_reason=done_reason,
        total_duration=int(total_time * 1e9),
        load_duration=stats_collector.load_duration,
        prompt_eval_count=len(prompt_tokens),
        eval_count=stats_collector.total_tokens,
        logprobs=logprobs,
    )
    if stats_collector.first_token_time:
        prompt_time = stats_collector.first_token_time - stats_collector.start_time
        response.prompt_eval_duration = int(prompt_time * 1e9)
        response.eval_duration = int((total_time - prompt_time) * 1e9)
    return response


async def generate_chat_output(
    generator,
    stats_collector: GenerationStatsCollector,
    request: ChatRequest,
    prompt_tokens: list[int],
) -> ChatResponse:
    """
    Collect full generation output and return an Ollama-compatible ChatResponse.
    Handles thinking blocks, tool call parsing, and optional logprobs.
    """
    result_text = ""
    generation_result = None
    all_tokens = []
    all_top_logprobs = []

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

        if request.logprobs and generation_result.top_logprobs:
            all_tokens.extend(generation_result.tokens)
            all_top_logprobs.extend(generation_result.top_logprobs)

    # Extract <think> blocks if thinking is enabled
    if request.think:
        re_think = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        result_text = re_think.sub("", result_text).strip()

    # Determine finish reason
    done_reason = "stop"
    if generation_result and generation_result.stop_condition:
        done_reason = generation_result.stop_condition.stop_reason

    # Parse tool calls from buffered output
    tool_calls_out = None
    content_out = result_text
    if any(p in result_text for p in TOOL_CALL_PREFIXES):
        tool_blocks, remaining = parse_tool_calls(result_text)
        if tool_blocks:
            done_reason = "tool_calls"
            content_out = remaining or ""
            tool_calls_out = [
                OllamaToolCall(
                    function=OllamaToolCallFunction(
                        name=tc.name,
                        arguments=tc.input if isinstance(tc.input, dict) else json.loads(tc.input),
                    )
                )
                for tc in tool_blocks
            ]

    # Build logprobs list
    logprobs = None
    if request.logprobs and all_tokens and all_top_logprobs:
        logprobs = build_logprobs(all_tokens, all_top_logprobs, request.top_logprobs or 1)

    return build_chat_response(
        model=request.model,
        text=content_out,
        tool_calls_out=tool_calls_out,
        logprobs=logprobs,
        done_reason=done_reason,
        stats_collector=stats_collector,
        prompt_tokens=prompt_tokens,
    )


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
        tool_blocks, remaining_content = parse_tool_calls(full_text)
        if tool_blocks:
            done_reason = "tool_calls"
            content_out = remaining_content or ""
            tool_calls_out = [
                OllamaToolCall(
                    function=OllamaToolCallFunction(
                        name=tc.name,
                        arguments=tc.input if isinstance(tc.input, dict) else json.loads(tc.input),
                    )
                )
                for tc in tool_blocks
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
