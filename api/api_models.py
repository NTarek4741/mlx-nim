import os
from typing import Annotated, Any, Literal, Union
from pydantic import AfterValidator, BaseModel, Field


# =============================================================================
# Model Validation
# =============================================================================


def model_exists(model: str) -> str:
    """Validate that a model exists in the models directory."""
    model_name = model.split(":")[0] if ":" in model else model

    if model_name.startswith("models/"):
        model_path = model_name
    else:
        model_path = os.path.join("./models", model_name)

    if not os.path.exists(model_path):
        raise ValueError(f"Model does not exist: {model}")
    return model_name


# =============================================================================
# SHARED MODELS (used by both Ollama and OpenAI endpoints)
# =============================================================================


class ResponseFormatText(BaseModel):
    """Plain text response format."""
    type: Literal["text"] = "text"


class ResponseFormatJSON(BaseModel):
    """JSON object response format."""
    type: Literal["json_object"] = "json_object"


class JSONSchema(BaseModel):
    """JSON schema definition."""
    name: str
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class ResponseFormatJSONSchema(BaseModel):
    """JSON schema response format."""
    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchema


ResponseFormat = Union[ResponseFormatText, ResponseFormatJSON, ResponseFormatJSONSchema]


class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""
    type: Literal["object"] = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None


class FunctionDefinition(BaseModel):
    """Function definition for tools."""
    name: str = Field(description="The name of the function")
    description: str | None = Field(default=None, description="Description of what the function does")
    parameters: FunctionParameters | None = Field(default=None, description="JSON schema for parameters")


class Tool(BaseModel):
    """Tool definition."""
    type: Literal["function"] = Field(
        default="function", description="Type of tool (always function)"
    )
    function: FunctionDefinition = Field(description="Function definition")


# =============================================================================
# OLLAMA MODELS
# =============================================================================


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str = Field(description="Model name/path")
    model: str = Field(description="Full model identifier")


class TagsResponse(BaseModel):
    """Response body for the /api/tags endpoint."""
    models: list[ModelInfo] = Field(description="List of available models")


class RunningModelInfo(BaseModel):
    """Information about a currently running/loaded model."""
    model: str = Field(description="Model name/path")


class PSResponse(BaseModel):
    """Response body for the /api/ps endpoint."""
    models: list[RunningModelInfo] = Field(description="List of currently running models")


class GenerationOptions(BaseModel):
    """Runtime options that control text generation."""

    seed: int | None = Field(
        default=None, description="Random seed used for reproducible outputs"
    )
    temperature: float | None = Field(
        default=0.15,
        description="Controls randomness in generation (higher = more random)",
    )
    top_k: int | None = Field(
        default=None, description="Limits next token selection to the K most likely"
    )
    top_p: float | None = Field(
        default=None,
        description="Cumulative probability threshold for nucleus sampling",
    )
    min_p: float | None = Field(
        default=None, description="Minimum probability threshold for token selection"
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequences that will halt generation"
    )
    num_ctx: int | None = Field(
        default=None, description="Context length size (number of tokens)"
    )
    num_predict: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    kv_bits: int | None = Field(
        default=None,
        ge=3,
        le=8,
        description="Number of bits for KV cache quantization. Must be between 3 and 8",
    )
    kv_group_size: int | None = Field(
        default=None, description="Group size for KV cache quantization"
    )
    quantized_kv_start: int | None = Field(
        default=None,
        description="When kv_bits is set, start quantizing the KV cache from this step onwards",
    )
    draft_model: str | None = Field(
        default=None,
        description="The name or path to the draft model for speculative decoding",
    )
    num_draft_tokens: int | None = Field(
        default=None,
        description="Number of tokens to draft when using speculative decoding",
    )


class TopLogprobEntry(BaseModel):
    token: str = Field(description="The text representation of the token")
    logprob: float = Field(description="The log probability of this token")
    bytes: list[int] | None = Field(
        default=None, description="The raw byte representation of the token"
    )


class LogprobEntry(BaseModel):
    token: str = Field(description="The text representation of the token")
    logprob: float = Field(description="The log probability of this token")
    bytes: list[int] | None = Field(
        default=None, description="The raw byte representation of the token"
    )
    top_logprobs: list[TopLogprobEntry] | None = Field(
        default=None,
        description="Most likely tokens and their log probabilities at this position",
    )


class GenerateResponse(BaseModel):
    model: str = Field(description="Model name")
    created_at: str = Field(description="ISO 8601 timestamp of response creation")
    response: str = Field(default="", description="The model's generated text response")
    thinking: str | None = Field(
        default=None, description="The model's generated thinking output"
    )
    done: bool = Field(
        default=False, description="Indicates whether generation has finished"
    )
    done_reason: str | None = Field(
        default=None, description="Reason the generation stopped"
    )

    total_duration: int | None = Field(
        default=None, description="Time spent generating the response in nanoseconds"
    )
    load_duration: int | None = Field(
        default=None, description="Time spent loading the model in nanoseconds"
    )
    prompt_eval_count: int | None = Field(
        default=None, description="Number of input tokens in the prompt"
    )
    prompt_eval_duration: int | None = Field(
        default=None, description="Time spent evaluating the prompt in nanoseconds"
    )
    eval_count: int | None = Field(
        default=None, description="Number of output tokens generated in the response"
    )
    eval_duration: int | None = Field(
        default=None, description="Time spent generating tokens in nanoseconds"
    )
    logprobs: list[LogprobEntry] | None = Field(
        default=None,
        description="Log probability information for the generated tokens when logprobs are enabled",
    )
    context: list[int] | None = Field(
        default=None,
        description="An encoding of the conversation used in this response, can be sent in the next request to keep conversational memory",
    )


class OllamaToolCallFunction(BaseModel):
    name: str = Field(description="Name of the function to call")
    arguments: dict = Field(description="Arguments to pass to the function")


class OllamaToolCall(BaseModel):
    id: str | None = Field(
        default=None, description="Unique identifier for the tool call"
    )
    type: Literal["function"] = Field(
        default="function", description="Type of tool call"
    )
    function: OllamaToolCallFunction = Field(description="Function call details")


class OllamaMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Author of the message"
    )
    content: str | list[dict] = Field(description="Message text content")
    images: list[str] | None = Field(
        default=None,
        description="Optional list of inline images for multimodal models (Base64-encoded image content)",
    )
    tool_calls: list[OllamaToolCall] | None = Field(
        default=None, description="Tool call requests produced by the model"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call this message responds to (for tool role messages)"
    )


class ChatRequest(BaseModel):
    """Request body model for Ollama-compatible chat endpoint."""

    # Required fields
    model: str = Field(description="Model name")
    messages: list[OllamaMessage] = Field(
        description="Chat history as an array of message objects (each with a role and content)"
    )

    # Optional fields
    tools: list[Tool] | None = Field(
        default=None,
        description="Optional list of function tools the model may call during the chat",
    )
    format: str | dict | None = Field(
        default=None,
        description='Format to return a response in. Can be "json" or a JSON schema',
    )
    options: GenerationOptions | None = Field(
        default=None, description="Runtime options that control text generation"
    )
    stream: bool = Field(default=True, description="Stream the response")
    think: bool | Literal["high", "medium", "low"] = Field(
        default=False,
        description='When true, returns separate thinking output in addition to content. Can be a boolean (true/false) or a string ("high", "medium", "low") for supported models.',
    )
    logprobs: bool = Field(
        default=False,
        description="Whether to return log probabilities of the output tokens",
    )
    top_logprobs: int = Field(
        default=0,
        description="Number of most likely tokens to return at each token position when logprobs are enabled",
    )


class OllamaResponseMessage(BaseModel):
    """Message object in chat response."""

    role: Literal["assistant", "tool"] = Field(description="Role of the message author")
    content: str = Field(default="", description="Message text content")
    tool_calls: list[OllamaToolCall] | None = Field(
        default=None, description="Tool calls made by the model"
    )


class ChatResponse(BaseModel):
    """Response body model for Ollama-compatible chat endpoint."""

    model: str = Field(description="Model name used to generate this message")
    created_at: str = Field(description="Timestamp of response creation (ISO 8601)")
    message: OllamaResponseMessage = Field(description="The generated message")
    done: bool = Field(
        default=False, description="Indicates whether the chat response has finished"
    )
    done_reason: str | None = Field(
        default=None, description="Reason the response finished"
    )

    total_duration: int | None = Field(
        default=None, description="Total time spent generating in nanoseconds"
    )
    load_duration: int | None = Field(
        default=None, description="Time spent loading the model in nanoseconds"
    )
    prompt_eval_count: int | None = Field(
        default=None, description="Number of tokens in the prompt"
    )
    prompt_eval_duration: int | None = Field(
        default=None, description="Time spent evaluating the prompt in nanoseconds"
    )
    eval_count: int | None = Field(
        default=None, description="Number of tokens generated in the response"
    )
    eval_duration: int | None = Field(
        default=None, description="Time spent generating tokens in nanoseconds"
    )
    logprobs: list[LogprobEntry] | None = Field(
        default=None,
        description="Log probability information for the generated tokens when logprobs are enabled",
    )


# =============================================================================
# OPENAI MODELS
# =============================================================================


# --- Content types ---

class TextContent(BaseModel):
    """Text content in a message."""
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    """Image URL reference."""
    url: str = Field(description="URL or base64 data URI of the image")
    detail: Literal["auto", "low", "high"] | None = Field(
        default="auto",
        description="Image detail level"
    )


class ImageContent(BaseModel):
    """Image content in a message."""
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[TextContent, ImageContent]


# --- Tool calls (OpenAI-specific) ---

class OpenAIFunctionCall(BaseModel):
    """Function call details."""
    name: str = Field(description="Name of the function to call")
    arguments: str = Field(description="JSON string of arguments")


class OpenAIToolCall(BaseModel):
    """Tool call made by the model."""
    id: str = Field(description="Unique ID for this tool call")
    index: int = Field(default=0, description="Index of this tool call")
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


# --- Messages (OpenAI-specific) ---

class OpenAISystemMessage(BaseModel):
    """System message."""
    role: Literal["system"]
    content: str


class OpenAIUserMessage(BaseModel):
    """User message with text or multimodal content."""
    role: Literal["user"]
    content: str | list[ContentPart]


class OpenAIAssistantMessage(BaseModel):
    """Assistant message, possibly with tool calls."""
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIToolMessage(BaseModel):
    """Tool result message."""
    role: Literal["tool"]
    content: str
    tool_call_id: str = Field(description="ID of the tool call this responds to")


OpenAIMessage = Union[OpenAISystemMessage, OpenAIUserMessage, OpenAIAssistantMessage, OpenAIToolMessage]


# --- Stream options ---

class StreamOptions(BaseModel):
    """Options for streaming responses."""
    include_usage: bool | None = Field(
        default=False,
        description="Include usage stats in final chunk"
    )


# --- Chat Completion Request ---

class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions endpoint."""

    # Required fields
    model: Annotated[str, AfterValidator(model_exists)] = Field(
        description="Model ID to use for completion"
    )
    messages: list[OpenAIMessage] = Field(
        description="List of messages in the conversation"
    )

    # Sampling parameters
    temperature: float | None = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)"
    )
    top_p: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )

    # Token limits
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate"
    )

    # Streaming
    stream: bool | None = Field(
        default=False,
        description="Enable streaming response"
    )
    stream_options: StreamOptions | None = Field(
        default=None,
        description="Streaming options"
    )

    # Stop sequences
    stop: str | list[str] | None = Field(
        default=None,
        description="Stop sequences (up to 4)"
    )

    # Tools
    tools: list[Tool] | None = Field(
        default=None,
        description="List of tools the model can call"
    )

    # Penalties
    frequency_penalty: float | None = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty (-2 to 2)"
    )

    # Response format
    response_format: ResponseFormat | None = Field(
        default=None,
        description="Response format specification"
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        description="Seed for deterministic generation"
    )

    # Log probabilities
    logprobs: bool | None = Field(
        default=None,
        description="Whether to return log probabilities of the output tokens"
    )
    top_logprobs: int | None = Field(
        default=None,
        description="Number of most likely tokens to return at each position (requires logprobs=true)"
    )


# --- Chat Completion Response (Non-Streaming) ---

class OpenAIResponseMessage(BaseModel):
    """Message in a completion response."""
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class ChoiceLogprobs(BaseModel):
    """Log probability information for a choice."""
    content: list["LogprobEntry"] | None = None


class Choice(BaseModel):
    """A completion choice."""
    index: int = Field(description="Index of this choice")
    message: OpenAIResponseMessage = Field(description="The generated message")
    finish_reason: Literal["stop", "length", "tool_calls"] | None = Field(
        default=None,
        description="Why generation stopped"
    )
    logprobs: ChoiceLogprobs | None = Field(
        default=None,
        description="Log probability information for this choice"
    )


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(description="Tokens in the prompt")
    completion_tokens: int = Field(description="Tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """Response from /v1/chat/completions (non-streaming)."""
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    choices: list[Choice] = Field(description="Completion choices")
    usage: Usage = Field(description="Token usage stats")


# --- Chat Completion Chunk (Streaming) ---

class DeltaMessage(BaseModel):
    """Delta content in a streaming chunk."""
    role: Literal["assistant"] = "assistant"
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class ChunkChoice(BaseModel):
    """A choice in a streaming chunk."""
    index: int = Field(description="Index of this choice")
    delta: DeltaMessage = Field(description="Delta content")
    finish_reason: Literal["stop", "length", "tool_calls"] | None = Field(
        default=None,
        description="Why generation stopped (null until final)"
    )


class ChatCompletionChunk(BaseModel):
    """A streaming chunk from /v1/chat/completions."""
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    choices: list[ChunkChoice] = Field(description="Chunk choices")
    usage: Usage | None = Field(
        default=None,
        description="Usage stats (only in final chunk if requested)"
    )


# --- Models List Response ---

class ModelObject(BaseModel):
    """A model object for /v1/models endpoint."""
    id: str = Field(description="Model identifier")
    object: Literal["model"] = "model"
    created: int = Field(description="Unix timestamp")
    owned_by: str = Field(default="local", description="Owner")


class ModelsListResponse(BaseModel):
    """Response from /v1/models endpoint."""
    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(description="List of models")


# =============================================================================
# ANTHROPIC MODELS
# =============================================================================

# --- Image Source Types ---

class Base64ImageSource(BaseModel):
    """Base64-encoded image source."""

    data: str = Field(description="Base64-encoded image data")
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        description="MIME type of the image"
    )
    type: Literal["base64"]


class URLImageSource(BaseModel):
    """URL-based image source."""

    type: Literal["url"]
    url: str = Field(description="URL of the image")


ImageSource = Union[Base64ImageSource, URLImageSource]


# --- Document Source Types ---

class Base64PDFSource(BaseModel):
    """Base64-encoded PDF source."""

    data: str = Field(description="Base64-encoded PDF data")
    media_type: Literal["application/pdf"] = Field(description="MIME type")
    type: Literal["base64"]


class PlainTextSource(BaseModel):
    """Plain text document source."""

    data: str = Field(description="Plain text content")
    media_type: Literal["text/plain"] = Field(description="MIME type")
    type: Literal["text"]


class ContentBlockSource(BaseModel):
    """Content block-based document source."""

    content: str | list["TextBlockParam"] = Field(
        description="Content as string or array of text blocks"
    )
    type: Literal["content"]


class URLPDFSource(BaseModel):
    """URL-based PDF source."""

    type: Literal["url"]
    url: str = Field(description="URL of the PDF")


DocumentSource = Union[
    Base64PDFSource, PlainTextSource, ContentBlockSource, URLPDFSource
]


# --- Content Block Params ---

class TextBlockParam(BaseModel):
    """Text content block."""

    text: str = Field(description="Text content")
    type: Literal["text"]
    cache_control: dict | None = Field(default=None)


class ImageBlockParam(BaseModel):
    """Image content block."""

    source: ImageSource = Field(description="Image source")
    type: Literal["image"]


class DocumentBlockParam(BaseModel):
    """Document content block."""

    source: DocumentSource = Field(description="Document source")
    type: Literal["document"]
    context: str | None = Field(
        default=None, description="Additional context for the document"
    )
    title: str | None = Field(default=None, description="Title of the document")


class ToolUseBlockParam(BaseModel):
    """Tool use block for requesting tool execution."""

    id: str = Field(description="Unique identifier for this tool use")
    input: dict[str, Any] = Field(description="Input parameters for the tool")
    name: str = Field(description="Name of the tool to use")
    type: Literal["tool_use"]


class ToolResultBlockParam(BaseModel):
    """Tool result block containing the result of a tool execution."""

    tool_use_id: str = Field(description="ID of the tool use this is responding to")
    type: Literal["tool_result"]
    content: (
        str
        | list[
            Union[
                TextBlockParam,
                ImageBlockParam,
                DocumentBlockParam,
            ]
        ]
        | None
    ) = Field(default=None, description="Result content from the tool execution")
    is_error: bool | None = Field(
        default=None, description="Whether the tool execution resulted in an error"
    )


class TranslateTextBlockParam(BaseModel):
    """Translation content block."""

    type: Literal["text"] = Field(description="Text content")
    source_lang_code: str = Field(description="Source language code")
    target_lang_code: str = Field(description="Target language code")
    text: str = Field(description="Text content")


class TranslateImageBlockParam(BaseModel):
    """Translation content block."""

    type: Literal["image"] = Field(description="Text content")
    source_lang_code: str = Field(description="Source language code")
    target_lang_code: str = Field(description="Target language code")
    image: ImageSource = Field(description="Image source")


ContentBlockParam = Union[
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    TranslateTextBlockParam,
    TranslateImageBlockParam,
]


# --- Message Param ---

class MessageParam(BaseModel):
    """Message in an Anthropic conversation."""

    role: Literal["user", "assistant"] = Field(
        description="The role of the message's author"
    )
    content: str | list[ContentBlockParam] = Field(description="The message content")


# --- Tool Choice ---

class ToolChoiceAuto(BaseModel):
    """The model will automatically decide whether to use tools."""

    type: Literal["auto"]
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use"
    )


class ToolChoiceAny(BaseModel):
    """The model will use any available tools."""

    type: Literal["any"]
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use"
    )


class ToolChoiceTool(BaseModel):
    """The model will use the specified tool."""

    type: Literal["tool"]
    name: str = Field(description="The name of the tool to use")
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use"
    )


class ToolChoiceNone(BaseModel):
    """The model will not be allowed to use tools."""

    type: Literal["none"]


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone]


# --- Anthropic Tool Definitions ---

class AnthropicToolInputSchema(BaseModel):
    """JSON schema for Anthropic tool input."""

    type: Literal["object"]
    properties: dict[str, Any] | None = Field(
        default=None, description="Properties of the input schema"
    )
    required: list[str] | None = Field(default=None, description="Required properties")


class AnthropicTool(BaseModel):
    """Anthropic tool definition."""

    name: str = Field(min_length=1, max_length=128, description="Name of the tool")
    input_schema: AnthropicToolInputSchema = Field(
        description="JSON schema for this tool's input"
    )
    description: str | None = Field(
        default=None, description="Description of what this tool does"
    )
    type: Literal["custom"] | None = Field(default=None, description="Tool type")


# --- Anthropic Messages Request ---

class MessagesParams(BaseModel):
    max_tokens: int = Field(ge=1, description="Maximum number of tokens to generate")
    model: Annotated[str, AfterValidator(model_exists)] = Field(
        description="The file system path to the model"
    )
    messages: list[MessageParam] = Field(
        description="A list of messages comprising the conversation so far."
    )

    # Core Sampling Parameters
    temperature: float | None = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0-1.0)",
    )
    max_completion_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    stream: bool | None = Field(
        default=False, description="Enable streaming of the response"
    )
    stop_sequences: list[str] | None = Field(
        default=None, description="Custom text sequences that will stop generation"
    )
    top_logprobs: int | None = Field(
        default=0, description="Number of top logprobs to return"
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Only sample from the top K options for each token",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Use nucleus sampling with this cumulative probability",
    )

    # System Prompt
    system: str | list[TextBlockParam] | None = Field(
        default=None, description="System prompt for context and instructions"
    )

    # Tools
    tools: list[AnthropicTool] | None = Field(
        default=None, description="Definitions of tools that the model may use"
    )
    tool_choice: ToolChoice | None = Field(
        default=None, description="How the model should use the provided tools"
    )

    # MLX-specific Parameters
    max_kv_size: int | None = Field(
        default=None, description="Max context size of the model"
    )
    kv_bits: int | None = Field(
        default=None,
        ge=3,
        le=8,
        description="Number of bits for KV cache quantization. Must be between 3 and 8",
    )
    kv_group_size: int | None = Field(
        default=None, description="Group size for KV cache quantization"
    )
    quantized_kv_start: int | None = Field(
        default=None,
        description="When --kv-bits is set, start quantizing the KV cache from this step onwards",
    )
    draft_model: str | None = Field(
        default=None,
        description="The file system path to the draft model for speculative decoding",
    )
    num_draft_tokens: int | None = Field(
        default=None,
        description="Number of tokens to draft when using speculative decoding",
    )
    print_prompt_progress: bool | None = Field(
        default=False, description="Enable printed prompt processing progress callback"
    )
    max_img_size: int | None = Field(
        default=None, description="Downscale images to this side length (px)"
    )
    json_schema: str | None = Field(
        default=None, description="JSON schema for the response"
    )


# --- Anthropic Response Models (native format) ---


class AnthropicUsage(BaseModel):
    """Anthropic token usage statistics (native format)."""

    input_tokens: int = Field(description="Number of tokens in the prompt")
    output_tokens: int = Field(
        description="Number of tokens in the generated completion"
    )


class AnthropicTextBlock(BaseModel):
    """Text content block in Anthropic response."""

    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseBlock(BaseModel):
    """Tool use content block in Anthropic response."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class AnthropicThinkingBlock(BaseModel):
    """Thinking content block in Anthropic response (extended thinking)."""

    type: Literal["thinking"] = "thinking"
    thinking: str


class AnthropicMessageResponse(BaseModel):
    """Native Anthropic Message response."""

    id: str = Field(description="Unique message identifier")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicThinkingBlock | AnthropicTextBlock | AnthropicToolUseBlock] = Field(
        description="Response content blocks"
    )
    model: str = Field(description="Model used")
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = Field(default=None, description="Reason generation stopped")
    stop_sequence: str | None = Field(
        default=None, description="Stop sequence that triggered stop, if any"
    )
    usage: AnthropicUsage = Field(description="Token usage stats")
