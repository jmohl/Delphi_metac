"""
LLM wrapper abstractions for forecast_bot_v2.

Provides a minimal replacement for forecasting_tools.GeneralLlm and
structure_output without introducing a heavy dependency layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Callable, Type

from openai import AsyncOpenAI

try:
    from anthropic import AsyncAnthropic  # type: ignore
except Exception:  # pragma: no cover - anthropic is optional
    AsyncAnthropic = None  # type: ignore


logger = logging.getLogger(__name__)


class GeneralLlm:
    """
    Lightweight wrapper around OpenAI/Anthropic async clients.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt

    async def invoke(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Execute a completion call with the configured model.

        Args:
            prompt: The prompt to send to the model
            tools: Optional tools/functions for function calling
            max_retries: Maximum number of retry attempts on failure (default: 3)
            reasoning_effort: Optional reasoning effort level for o1/GPT-5.1 models
                             (e.g., "low", "medium", "high")

        Returns:
            The model's response as a string

        Raises:
            ValueError: If API credentials are missing
            Exception: If all retry attempts fail
        """
        # Validate credentials before making API calls
        valid, reason = validate_model_credentials(self.model)
        if not valid:
            raise ValueError(f"Missing API credentials: {reason}")

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(max_retries):
            try:
                if _is_openrouter_model(self.model):
                    return await _invoke_openai(
                        model=_strip_openrouter_prefix(self.model),
                        prompt=prompt,
                        tools=tools,
                        system=self.system_prompt,
                        reasoning_effort=reasoning_effort,
                        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                        extra_headers=_openrouter_headers(),
                    )

                if _is_anthropic_model(self.model):
                    return await _invoke_anthropic(
                        model=self.model,
                        prompt=prompt,
                        tools=tools,
                        system=self.system_prompt,
                    )

                # Default to OpenAI client
                return await _invoke_openai(
                    model=self.model,
                    prompt=prompt,
                    tools=tools,
                    system=self.system_prompt,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    wait_time = 2 ** attempt
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")

        # If we get here, all retries failed
        raise last_exception  # type: ignore[misc]


async def _invoke_openai(
    model: str,
    prompt: str,
    tools: list[dict[str, Any]] | None,
    system: str | None,
    *,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """
    Invoke OpenAI or OpenRouter using the appropriate API.

    Args:
        model: Model name to use
        prompt: User prompt
        tools: Optional list of tool definitions
        system: Optional system prompt
        base_url: Optional base URL (for OpenRouter)
        extra_headers: Optional extra headers (for OpenRouter)
        reasoning_effort: Optional reasoning effort level for o1/GPT-5.1 models

    Returns:
        The model's text response
    """
    # Select appropriate API key based on whether we're using OpenRouter
    if base_url:
        api_key = os.getenv("OPENROUTER_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)

    # Check if web_search tools are requested - requires responses API
    has_web_search = tools and any(tool.get("type") == "web_search" for tool in tools)

    if has_web_search and not base_url:
        # Use the responses.create() API for web search (OpenAI only, not OpenRouter)
        # Combine system prompt and user prompt into input
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        kwargs: dict[str, Any] = {
            "model": model,
            "input": full_prompt,
            "tools": tools,
        }
        if reasoning_effort is not None and not base_url:
            kwargs["reasoning"] = {"effort": reasoning_effort if reasoning_effort is not None else None,}

        response = await client.responses.create(**kwargs)  # type: ignore[attr-defined]
        return (response.output_text or "").strip()  # type: ignore[attr-defined]
    else:
        # Use the standard chat.completions API
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools and not has_web_search:
            kwargs["tools"] = tools

        # Add reasoning_effort for reasoning models (o1, gpt-5.x)
        # Only when NOT using OpenRouter (they may not support it)
        if reasoning_effort is not None and not base_url:
            kwargs["reasoning"] = {"effort": reasoning_effort if reasoning_effort is not None else None,}

        response = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

        # Extract text from response
        message_content = response.choices[0].message.content
        return (message_content or "").strip()


async def _invoke_anthropic(
    model: str,
    prompt: str,
    tools: list[dict[str, Any]] | None,
    system: str | None,
) -> str:
    if AsyncAnthropic is None:
        raise RuntimeError("Anthropic client is not available. Install anthropic or use an OpenAI model.")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = AsyncAnthropic(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]

    # Build kwargs, only include optional parameters if provided
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    resp = await client.messages.create(**kwargs)  # type: ignore[arg-type]

    # Anthropic returns a content array; concatenate text blocks
    pieces: list[str] = []
    for block in getattr(resp, "content", []) or []:
        if hasattr(block, "text"):
            pieces.append(block.text)  # type: ignore[attr-defined]
    return "\n".join(pieces)


def structure_output(raw_text: str, model: Type, *, postprocess: Callable[[Any], Any] | None = None) -> Any:
    """
    Minimal replacement for forecasting_tools.structure_output.

    Attempts to parse JSON from the LLM output and validate using the provided class
    (Pydantic-style or simple dataclass with **kwargs). If validation fails, raises ValueError.
    """
    json_obj = _extract_json(raw_text)
    if json_obj is None:
        raise ValueError("Could not parse JSON from model output.")

    try:
        parsed = model(**json_obj)  # type: ignore[arg-type]
    except Exception:
        # Try the fallback of passing the raw dict if model is not callable
        parsed = model(json_obj)  # type: ignore[misc]

    if postprocess:
        return postprocess(parsed)
    return parsed


def _extract_json(text: str) -> Any | None:
    """
    Extract the first JSON object/array from text.

    Tries multiple strategies:
    1. Parse the entire text as JSON
    2. Extract JSON between ```json and ``` code blocks
    3. Find the largest balanced {...} or [...] block
    """
    # Strategy 1: Try parsing the entire text
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strategy 2: Extract from code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except Exception:
            pass

    # Strategy 3: Find balanced braces (handles nested objects)
    # Look for the largest complete JSON object
    for pattern in [r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]']:
        matches = re.finditer(pattern, text, flags=re.DOTALL)
        # Try matches from longest to shortest
        for match in sorted(matches, key=lambda m: len(m.group(0)), reverse=True):
            try:
                return json.loads(match.group(0))
            except Exception:
                continue

    return None


def _is_anthropic_model(model: str) -> bool:
    return "claude" in model or model.startswith("anthropic/")


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("openrouter/")


def validate_model_credentials(model: str) -> tuple[bool, str | None]:
    """
    Ensure the appropriate API key exists for the requested model.

    Returns:
        (is_valid, reason_if_invalid)
    """
    if _is_openrouter_model(model):
        if not os.getenv("OPENROUTER_API_KEY"):
            return False, "OPENROUTER_API_KEY is not set for OpenRouter models"
        return True, None

    if _is_anthropic_model(model):
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set for Anthropic models"
        return True, None

    # Assume OpenAI if nothing else matches
    if not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is not set for OpenAI models"
    return True, None


def _strip_openrouter_prefix(model: str) -> str:
    """
    Remove leading 'openrouter/' to pass provider:variant to OpenRouter API.
    """
    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    return model


def _openrouter_headers() -> dict[str, str]:
    """
    Optional headers recommended by OpenRouter.
    """
    headers: dict[str, str] = {}
    ref = os.getenv("OPENROUTER_REFERRER")
    title = os.getenv("OPENROUTER_TITLE")
    if ref:
        headers["HTTP-Referer"] = ref
    if title:
        headers["X-Title"] = title
    return headers or None  # type: ignore[return-value]


__all__ = ["GeneralLlm", "structure_output", "validate_model_credentials"]
