import gc
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated

import mlx.core as mx
from fastapi import Body, FastAPI, Query
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from mlx_lm import convert as mlx_lm_convert
from mlx_vlm import convert as mlx_vlm_convert
from starlette.responses import StreamingResponse

# Import API examples
from api.api_examples import (
    CHAT_COMPLETIONS_EXAMPLES,
    CHAT_EXAMPLES,
    CREATE_MODEL_OUTPUT_DIR_EXAMPLES,
    CREATE_MODEL_REPO_ID_EXAMPLES,
    MESSAGES_EXAMPLES,
    PULL_MODEL_REPO_ID_EXAMPLES,
)
from api.api_models import (
    ChatCompletionRequest,
    ChatRequest,
    GenerationOptions,
    MessagesParams,
    ModelInfo,
    ModelObject,
    ModelsListResponse,
    PSResponse,
    RunningModelInfo,
    TagsResponse,
)

# Import utility functions
from api.api_utils import (
    GenerationStatsCollector,
    anthropic_stream,
    anthropic_to_chat_convert,
    chat_render,
    chat_stream,
    generate_anthropic_output,
    generate_openai_output,
    generate_output,
    load_and_cache_model,
    model_cache,
    openai_stream,
    openai_to_chat_convert,
)
from mlx_engine.generate import create_generator
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter

# Configure logging to match FastAPI/uvicorn style
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
)

app = FastAPI()
logger = logging.getLogger(__name__)


# POST - Generate a chat message
@app.post("/api/chat")
async def chat(request: Annotated[ChatRequest, Body(examples=CHAT_EXAMPLES)]):
    """ 
    Generate a chat response based on conversation messages.

    Description:
        Processes a chat conversation and generates a response using the specified model.
        Supports both streaming and non-streaming responses, tool calls, and image inputs.

    Args:
        request (ChatRequest): The chat request containing:
            - model: The model identifier to use
            - messages: List of conversation messages
            - options: Generation options (temperature, top_k, etc.)
            - stream: Whether to stream the response
            - tools: Optional tool definitions for function calling
            - format: Optional JSON schema for structured output

    Returns:
        StreamingResponse | dict: Either a streaming response for real-time generation
            or a dictionary containing the complete chat response with message content
            and generation statistics.
    """
    options = request.options or GenerationOptions()

    try:
        generator, stats_collector, prompt_tokens = await engine_core(
            options, request.model, request
        )
    except Exception as e:
        logger.info(f"Error Occured: {str(e)}")
        return {"error": f"Error Occured: {str(e)}"}

    try:
        if request.stream:
            return StreamingResponse(
                chat_stream(
                    generator,
                    request.model,
                    stats_collector,
                    prompt_tokens,
                    request.logprobs,
                    request.top_logprobs,
                )
            )
        else:
            return await generate_output(
                generator, stats_collector, request, prompt_tokens
            )
    except Exception as e:
        logger.info(f"Error while generating chat response: {str(e)}")
        raise e


async def engine_core(options, model_name, request):
    try:
        model_kit, load_duration_ns = load_and_cache_model(
            model=model_name,
            num_ctx=options.num_ctx,
            kv_bits=options.kv_bits,
            kv_group_size=options.kv_group_size,
            quantized_kv_start=options.quantized_kv_start,
            draft_model=options.draft_model,
        )
    except Exception as e:
        logger.info(f"Error while loading: {str(e)}")
        raise e

    try:
        images = []
        prompt_tokens = await chat_render(request.messages, request.tools, images)
    except Exception as e:
        logger.info(f"Error while rendering chat messages: {str(e)}")
        raise e

    stats_collector = GenerationStatsCollector()
    stats_collector.load_duration = load_duration_ns

    # Handle JSON schema validation
    json_schema = None
    if request.format:
        json_schema = json.dumps(request.format)

    try:
        generator = create_generator(
            model_kit,
            prompt_tokens,
            images_b64=images,
            stop_strings=options.stop,
            max_tokens=options.num_predict,
            top_logprobs=request.top_logprobs if request.logprobs else None,
            prompt_progress_reporter=LoggerReporter(),
            seed=options.seed,
            temp=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            min_p=options.min_p,
            json_schema=json_schema,
            num_draft_tokens=options.num_draft_tokens,
        )
    except Exception as e:
        logger.info(f"Error creating generator: {str(e)}")
        raise e
    return generator, stats_collector, prompt_tokens


# POST - Generate embeddings
@app.post("/api/embeddings")
async def embeddings():
    """
    Generate embeddings for input text.

    Description:
        Creates vector embeddings for the provided input text using the specified model.
        Currently returns an empty embeddings array as a placeholder.

    Args:
        None

    Returns:
        dict[str, list]: A dictionary containing:
            - embeddings: List of embedding vectors (currently empty placeholder)
    """
    try:
        return {"embeddings": []}
    except Exception as e:
        logger.info(f"Error generating embeddings: {str(e)}")
        return {"error": f"Error generating embeddings: {str(e)}"}


# =============================================================================
# Anthropic API Endpoint
# =============================================================================


@app.post("/v1/messages")
async def messages(params: Annotated[MessagesParams, Body(examples=MESSAGES_EXAMPLES)]):
    """
    Anthropic Messages API endpoint.

    Description:
        Processes messages using the Anthropic API format and generates responses.
        Converts Anthropic format to internal ChatRequest, runs through engine_core,
        then returns an Anthropic-format response.
        Supports both streaming and non-streaming modes.

    Args:
        params (MessagesParams): The message parameters containing:
            - model: The model identifier to use
            - messages: List of conversation messages
            - max_tokens: Maximum tokens to generate
            - stream: Whether to stream the response
            - system: Optional system prompt
            - temperature: Sampling temperature

    Returns:
        StreamingResponse | dict: Either a streaming response for real-time generation
            or a dictionary containing the complete message response with content,
            usage statistics, and stop reason.
    """
    try:
        # 1. Convert Anthropic params to ChatRequest
        request = anthropic_to_chat_convert(params)

        # 2. Run through engine_core (same pattern as OpenAI)
        generator, stats_collector, prompt_tokens = await engine_core(
            request.options, params.model, request
        )

        # 3. Return Anthropic-format response
        if params.stream:
            return StreamingResponse(
                anthropic_stream(
                    generator,
                    params.model,
                    stats_collector,
                    prompt_tokens,
                    params.top_logprobs > 0 if params.top_logprobs else False,
                    params.top_logprobs or 0,
                ),
                media_type="text/event-stream",
            )
        else:
            return await generate_anthropic_output(
                generator,
                stats_collector,
                params,
                len(prompt_tokens),
            )
    except Exception as e:
        logger.info(f"Error processing Anthropic messages request: {str(e)}")
        return {"error": f"Error processing messages request: {str(e)}"}


# =============================================================================
# OpenAI API Endpoints
# =============================================================================


@app.post("/v1/chat/completions")
async def chat_completions(
    params: Annotated[ChatCompletionRequest, Body(examples=CHAT_COMPLETIONS_EXAMPLES)],
):
    """
    OpenAI-compatible chat completions endpoint.

    Description:
        Processes chat completion requests using the OpenAI API format.
        Supports both streaming and non-streaming modes.

    Args:
        params (ChatCompletionRequest): The chat completion parameters containing:
            - model: The model identifier to use
            - messages: List of conversation messages
            - max_tokens: Maximum tokens to generate
            - stream: Whether to stream the response
            - temperature: Sampling temperature
            - top_p: Nucleus sampling parameter
            - n: Number of completions to generate

    Returns:
        StreamingResponse | dict: Either a streaming response for real-time generation
            or a dictionary containing the complete chat completion with choices,
            usage statistics, and finish reason.
    """
    request = openai_to_chat_convert(params)
    generator, stats_collector, prompt_tokens = await engine_core(
        request.options, params.model, request
    )

    # 8. Return streaming or non-streaming response
    if params.stream:
        include_usage = False
        if params.stream_options and params.stream_options.include_usage:
            include_usage = True

        return StreamingResponse(
            openai_stream(
                generator,
                params.model,
                stats_collector,
                prompt_tokens,
                include_usage,
            ),
            media_type="text/event-stream",
        )
    else:
        return await generate_openai_output(
            generator,
            stats_collector,
            params,
            len(prompt_tokens),
        )


# =============================================================================
# Convenience Endpoints
# =============================================================================


# GET - List models
@app.get("/api/tags")
async def list_models():
    """
    List all available models.

    Description:
        Scans the models directory and returns a list of all available models
        in Ollama-compatible format with organization/model:tag naming.

    Args:
        None

    Returns:
        TagsResponse | dict: A TagsResponse object containing a list of ModelInfo
            objects with model names and metadata, or an error dictionary if
            the operation fails.
    """
    try:
        models = []
        models_dir = "./models"

        for org in os.listdir(models_dir):
            org_path = os.path.join(models_dir, org)

            for model_name in os.listdir(org_path):
                model_path = os.path.join(org_path, model_name)

                # Use Ollama-style naming with :latest tag
                full_name = f"{org}/{model_name}:latest"

                model_info = ModelInfo(
                    name=full_name,
                    model=full_name,
                )
                models.append(model_info)

        models.sort(key=lambda m: m.name)
        return TagsResponse(models=models)
    except Exception as e:
        logger.info(f"Error listing models: {str(e)}")
        return {"error": f"Error listing models: {str(e)}"}


# GET - List running models
@app.get("/api/ps")
async def list_running_models():
    """
    List currently loaded/running models.

    Description:
        Returns information about models currently loaded in memory and ready
        for inference. Checks the model cache for active models.

    Args:
        None

    Returns:
        PSResponse | dict[str, str]: A PSResponse object containing a list of
            RunningModelInfo objects for each loaded model, or an error dictionary
            if the operation fails.
    """
    try:
        models = []

        # Check if a model is currently loaded
        if (
            model_cache.get("model_kit") is not None
            and model_cache.get("load_params") is not None
        ):
            load_params = model_cache["load_params"]
            model_path = load_params["model_path"]

            # Extract model name from path (e.g., "models/google/gemma-7b" -> "google/gemma-7b:latest")
            model_name = (
                model_path.replace("models/", "", 1)
                if model_path.startswith("models/")
                else model_path
            )
            model_name = f"{model_name}:latest"  # Add Ollama-style tag

            model_info = RunningModelInfo(model=model_name)
            models.append(model_info)
        return PSResponse(models=models)
    except Exception as e:
        logger.info(f"Error listing running models: {str(e)}")
        return {"error": f"Error listing running models: {str(e)}"}


# POST - Create a model
@app.post("/api/create")
async def create_model(
    repo_id: str = Query(
        description="HuggingFace repository ID", examples=CREATE_MODEL_REPO_ID_EXAMPLES
    ),
    output_dir: str | None = Query(
        default=None,
        description="Output directory for the converted model",
        examples=CREATE_MODEL_OUTPUT_DIR_EXAMPLES,
    ),
):
    """
    Create a model by converting from HuggingFace format.

    Description:
        Downloads and converts a model from HuggingFace to MLX format with
        4-bit quantization. Attempts VLM conversion first, then falls back
        to LM conversion if VLM fails.

    Args:
        repo_id (str): The HuggingFace repository ID (e.g., "organization/model-name")
        output_dir (str | None): Optional output directory for the converted model.
            Defaults to "./models/{repo_id}" if not specified.

    Returns:
        dict[str, str | dict[str, str]]: A dictionary containing:
            - status: "success" or "error"
            - message: Description of the result
            - output_path: Path where the model was saved (on success)
            - converter: Which converter was used (on success)
            - quantization: Quantization level used (on success)
            - details: Error details from both converters (on failure)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"./models/{repo_id}"

    vlm_error = None
    lm_error = None

    try:
        # Second attempt: Try mlx_vlm.convert with 4-bit quantization
        print(
            f"Attempting to convert {repo_id} using mlx_vlm.convert with 4-bit quantization..."
        )
        mlx_vlm_convert(
            hf_path=repo_id,
            mlx_path=output_dir,
            quantize=True,
            q_group_size=64,
            q_bits=4,
        )
        return {
            "status": "success",
            "message": f"Model {repo_id} successfully converted using mlx_vlm with 4-bit quantization",
            "output_path": output_dir,
            "converter": "mlx_vlm",
            "quantization": "4-bit",
        }
    except Exception as e:
        vlm_error = e
        print(f"mlx_vlm.convert failed: {str(vlm_error)}")
    try:
        # First attempt: Try mlx_lm.convert with 4-bit quantization
        print(
            f"Attempting to convert {repo_id} using mlx_lm.convert with 4-bit quantization..."
        )
        mlx_lm_convert(
            hf_path=repo_id,
            mlx_path=output_dir,
            quantize=True,
            q_group_size=64,
            q_bits=4,
        )
        return {
            "status": "success",
            "message": f"Model {repo_id} successfully converted using mlx_lm with 4-bit quantization",
            "output_path": output_dir,
            "converter": "mlx_lm",
            "quantization": "4-bit",
        }
    except Exception as e:
        lm_error = e
        print(f"mlx_lm.convert failed: {str(lm_error)}")

    # Both converters failed
    return {
        "status": "error",
        "message": "Model conversion failed",
        "details": {
            "mlx_lm_error": str(lm_error) if lm_error else "Not attempted/Unknown",
            "mlx_vlm_error": str(vlm_error) if vlm_error else "Not attempted/Unknown",
        },
    }


# POST - Pull a model
@app.post("/api/pull")
async def pull_model(
    repo_id: str = Query(
        description="HuggingFace repository ID to download",
        examples=PULL_MODEL_REPO_ID_EXAMPLES,
    ),
):
    """
    Pull a model from HuggingFace Hub.

    Description:
        Downloads a model from HuggingFace Hub to the local models directory
        using snapshot_download for efficient caching and resumable downloads.

    Args:
        repo_id (str): The HuggingFace repository ID (e.g., "organization/model-name")

    Returns:
        dict[str, str]: A dictionary containing:
            - status: "success" or "error"
            - message: Description of the result or error details
    """
    try:
        creator, model = repo_id.split("/")
        snapshot_download(repo_id=repo_id, local_dir=f"./models/{creator}/{model}")
        return {
            "status": "success",
            "message": f"Model {repo_id} downloaded successfully to ./models/{creator}/download-{model}",
        }
    except Exception as e:
        logger.info(f"Error pulling model: {str(e)}")
        return {"status": "error", "message": str(e)}


# DELETE - Delete a model
@app.delete("/api/delete")
async def delete_model(
    model: str = Query(
        description="Model name to delete (e.g., 'org/model-name' or 'org/model-name:latest')"
    ),
):
    """
    Delete a model.

    Description:
        Removes a model and its folder from the local models directory.
        Strips Ollama-style tags (e.g., ':latest') before resolving the path.

    Args:
        model (str): The model identifier (e.g., "organization/model-name")

    Returns:
        dict[str, str]: A dictionary containing the operation status and message.
    """
    try:
        # Strip Ollama-style tags
        model_name = model.split(":")[0] if ":" in model else model
        model_path = os.path.join("./models", model_name)

        if not os.path.exists(model_path):
            return {"status": "error", "message": f"Model not found at {model_path}"}

        # Unload from cache if this model is currently loaded
        if (
            model_cache.get("load_params")
            and model_cache["load_params"].get("model_path") == model_path
        ):
            model_cache.clear()
            gc.collect()

        shutil.rmtree(model_path)
        return {
            "status": "success",
            "message": f"Model '{model_name}' deleted from {model_path}",
        }
    except Exception as e:
        logger.info(f"Error deleting model: {str(e)}")
        return {"status": "error", "message": str(e)}


# DELETE - Clear HuggingFace cache
@app.delete("/api/clear-huggingface-cache")
async def clear_huggingface_cache():
    """
    Clear the HuggingFace Hub cache.

    Description:
        Deletes the contents of ~/.cache/huggingface/hub to free up disk space.

    Returns:
        dict[str, str]: A dictionary containing:
            - status: "success" or "error"
            - message: Description of the result
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            return {
                "status": "success",
                "message": f"HuggingFace cache cleared at {cache_dir}",
            }
        else:
            return {
                "status": "success",
                "message": "Cache directory does not exist, nothing to clear",
            }
    except Exception as e:
        logger.info(f"Error clearing HuggingFace cache: {str(e)}")
        return {"status": "error", "message": str(e)}


# GET - Get version
@app.get("/api/version")
async def get_version():
    """
    Get API version.

    Description:
        Returns the current version of the MLX Engine API.

    Args:
        None

    Returns:
        dict[str, str]: A dictionary containing the API version string.
    """
    return {"version": "0.1.0"}
