import asyncio
import gc
import json
import logging
import re
import time
from datetime import datetime, timezone

import mlx as mx

logger = logging.getLogger(__name__)

from api.api_models import (
    GenerateResponse,
    LogprobEntry,
    OllamaMessage,
    Tool,
    TopLogprobEntry,
)
from mlx_engine.generate import load_draft_model, load_model, tokenize
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
        logger.info(f"[LOAD] Model cached — {load_params['model_path']}")
        return model_cache.get("model_kit"), 0  # No load time for cached model
    else:
        if model_cache.get("model_kit") is not None:
            del model_cache["model_kit"]
            del model_cache["load_params"]
            mx.core.clear_cache()
            gc.collect()
            logger.info("[LOAD] Previous model cleared from cache")

    load_start_time = time.time()
    logger.info(f"[LOAD] Loading {load_params['model_path']}...")
    model_kit = load_model(
        load_params["model_path"],
        max_kv_size=load_params["max_kv_size"],
        max_seq_nums=1,
        trust_remote_code=False,
        kv_bits=load_params["kv_bits"],
        kv_group_size=load_params["kv_group_size"],
        quantized_kv_start=load_params["quantized_kv_start"],
    )

    # Load draft model if specified
    if load_params["draft_model"]:
        logger.info(f"[LOAD] Loading draft model: {load_params['draft_model']}...")
        load_draft_model(model_kit, load_params["draft_model"])
        logger.info("[LOAD] Draft model loaded")

    load_end_time = time.time()
    load_duration_ns = int((load_end_time - load_start_time) * 1e9)
    elapsed = load_end_time - load_start_time

    # Update cache
    model_cache["model_kit"] = model_kit
    model_cache["load_params"] = load_params

    logger.info(f"[LOAD] Model ready in {elapsed:.2f}s — {load_params['model_path']}")
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
            logger.warning("[STATS] No tokens generated")
            return

        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token
        tokens_per_second = (
            self.total_tokens / effective_time if effective_time > 0 else float("inf")
        )
        logger.info(f"[STATS] tokens/s={tokens_per_second:.2f} ttft={time_to_first_token:.2f}s total={self.total_tokens} elapsed={total_time:.2f}s"
                    + (f" draft_accepted={self.num_accepted_draft_tokens}" if self.num_accepted_draft_tokens is not None else ""))

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
                build_logprobs(
                    generation_result.tokens,
                    generation_result.top_logprobs,
                    getattr(generate_query, "top_logprobs", 1) or 1,
                )
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
    logger.info(f"[RENDER] Applying template — {len(messages)} messages, tools={tools is not None}, images={len(images)}")
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
            # No images - content is a string or structured list of blocks
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
    logger.debug(f"[RENDER] Prompt tail: {repr(prompt[-200:])}")
    tokens = tokenize(model_cache["model_kit"], prompt)
    logger.info(f"[RENDER] Tokenized → {len(tokens)} prompt tokens")
    return tokens

