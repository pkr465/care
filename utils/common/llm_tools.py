# llm_tools.py
"""
Generic multi-provider LLM toolkit for codebase analysis.

Provides:
- Provider-routed LLM calls (Anthropic Claude, QGenie, extensible)
- Automatic provider selection from "provider::model" config format
- Structured intent extraction from natural language prompts
- JSON extraction from LLM responses
- Markdown prompt template rendering
- Vector DB retrieval (optional, injected)
- Tool registry with schema introspection

Provider routing:
    "anthropic::claude-sonnet-4-20250514"  → AnthropicProvider (anthropic SDK)
    "qgenie::qwen2.5-14b-1m"              → QGenieProvider    (QGenieChat)
    "vertexai::gemini-2.5-pro"             → VertexAIProvider  (langchain stub)
    "azure::gpt-4.1"                       → AzureProvider     (langchain stub)

Dependencies: None required at import time. Provider SDKs are lazy-imported:
    anthropic: pip install anthropic
    qgenie:    pip install qgenie
    vertexai:  pip install langchain-google-vertexai
    azure:     pip install langchain-openai

Usage:
    from utils.common.llm_tools import LLMTools

    # Auto-routes to correct provider based on config
    tools = LLMTools()
    tools = LLMTools(model="anthropic::claude-sonnet-4-20250514")
    tools = LLMTools(model="qgenie::qwen2.5-14b-1m")

    response = tools.llm_call("Analyze this code for security issues...")
    intent   = tools.extract_intent_from_prompt("Compare module A and B")
"""

from __future__ import annotations

import abc
import json
import re
import uuid
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """Configuration for LLM provider and behavior."""

    # Provider / model — "provider::model" format (parsed automatically)
    raw_model: str = "anthropic::claude-sonnet-4-20250514"
    coding_model: str = "anthropic::claude-sonnet-4-20250514"

    # API keys (provider-specific)
    anthropic_api_key: Optional[str] = None
    qgenie_api_key: Optional[str] = None

    # Request defaults
    max_tokens: int = 8192
    temperature: float = 0.1
    timeout: int = 120          # seconds
    max_retries: int = 2

    # Intent extraction
    intent_max_tokens: int = 4096
    intent_temperature: float = 0.0

    # Token budget (for prompt truncation)
    max_prompt_tokens: int = 100_000

    # Prompt file path
    chat_prompt_file_path: Optional[str] = None

    @property
    def provider(self) -> str:
        """Extract provider name from raw_model string."""
        return parse_provider_model(self.raw_model)[0]

    @property
    def model(self) -> str:
        """Extract model name from raw_model string."""
        return parse_provider_model(self.raw_model)[1]

    @classmethod
    def from_env(cls, env_config=None) -> "LLMConfig":
        """Build config from EnvConfig, GlobalConfig, or environment variables."""
        import os

        if env_config is None:
            # Try GlobalConfig first, fall back to EnvConfig
            try:
                from utils.parsers.global_config_parser import GlobalConfig
                env_config = GlobalConfig()
            except Exception:
                try:
                    from utils.parsers.env_parser import EnvConfig
                    env_config = EnvConfig()
                except ImportError:
                    env_config = None

        def _get(key: str, default: Any = None) -> Any:
            if env_config and hasattr(env_config, "get"):
                val = env_config.get(key)
                if val is not None and val != "":
                    return val
            return os.getenv(key, default)

        raw_model = _get("llm.model") or _get("LLM_MODEL") or "anthropic::claude-sonnet-4-20250514"

        return cls(
            raw_model=str(raw_model),
            coding_model=str(_get("llm.coding_model") or _get("CODING_LLM_MODEL") or raw_model),
            anthropic_api_key=_get("llm.anthropic_api_key") or _get("ANTHROPIC_API_KEY"),
            qgenie_api_key=_get("llm.qgenie_api_key") or _get("QGENIE_API_KEY"),
            max_tokens=int(_get("llm.max_tokens") or _get("LLM_MAX_TOKENS") or 8192),
            temperature=float(_get("llm.temperature") or _get("LLM_TEMPERATURE") or 0.1),
            timeout=int(_get("llm.timeout") or _get("LLM_TIMEOUT") or 120),
            max_retries=int(_get("llm.max_retries") or 2),
            intent_max_tokens=int(_get("llm.intent_max_tokens") or 4096),
            intent_temperature=float(_get("llm.intent_temperature") or 0.0),
            max_prompt_tokens=int(_get("llm.max_prompt_tokens") or 100_000),
            chat_prompt_file_path=_get("paths.chat_prompt_file_path") or _get("CHAT_PROMPT_FILE_PATH"),
        )


def parse_provider_model(raw: str) -> tuple:
    """
    Parse a 'provider::model' string into (provider, model).

    Examples:
        "anthropic::claude-sonnet-4-20250514" → ("anthropic", "claude-sonnet-4-20250514")
        "qgenie::qwen2.5-14b-1m"              → ("qgenie", "qwen2.5-14b-1m")
        "claude-sonnet-4-20250514"             → ("anthropic", "claude-sonnet-4-20250514")
        "qwen2.5-14b-1m"                       → ("qgenie", "qwen2.5-14b-1m")

    Falls back to heuristics if no '::' separator is present.
    """
    raw = str(raw).strip()
    if "::" in raw:
        provider, model = raw.split("::", 1)
        return provider.strip().lower(), model.strip()

    # Heuristic: infer provider from model name
    lower = raw.lower()
    if any(k in lower for k in ("claude", "anthropic")):
        return "anthropic", raw
    elif any(k in lower for k in ("gemini", "palm", "vertex")):
        return "vertexai", raw
    elif any(k in lower for k in ("gpt", "azure", "openai")):
        return "azure", raw
    else:
        # Default to qgenie for unrecognized model names
        return "qgenie", raw


# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class LLMError(Exception):
    """Base error for LLM operations."""


class LLMProviderError(LLMError):
    """Error communicating with the LLM provider."""


class LLMResponseError(LLMError):
    """Error parsing or validating an LLM response."""


class IntentExtractionError(LLMError):
    """Failed to extract structured intent from a prompt."""


class ProviderNotAvailableError(LLMError):
    """Required provider SDK is not installed."""


# ═══════════════════════════════════════════════════════════════════════════
# Provider Abstraction Layer
# ═══════════════════════════════════════════════════════════════════════════

class BaseLLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.

    Each provider implements `complete()` to send messages and return
    the text response. Provider SDKs are lazy-imported to avoid hard
    dependencies.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @abc.abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a completion request and return the text response."""
        ...

    def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        tool_choice: Optional[Dict] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion with tool definitions (function calling).

        Default implementation: not all providers support this.
        Returns {"text": str, "tool_calls": list, "stop_reason": str}.
        """
        # Fallback: just do a regular completion
        text = self.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        )
        return {"text": text, "tool_calls": [], "stop_reason": "end_turn"}


# ---------------------------------------------------------------------------
# Anthropic Claude Provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.
    Uses the `anthropic` Python SDK for completions and tool calling.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderNotAvailableError(
                    "The 'anthropic' package is required for Claude models. "
                    "Install it with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(
                api_key=self.config.anthropic_api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if system:
            kwargs["system"] = system

        try:
            start = time.monotonic()
            response = self.client.messages.create(**kwargs)
            elapsed = time.monotonic() - start

            logger.debug(
                "Anthropic API: model=%s in=%s out=%s %.2fs",
                self.config.model,
                getattr(response.usage, "input_tokens", "?"),
                getattr(response.usage, "output_tokens", "?"),
                elapsed,
            )

            # Extract text from content blocks
            parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)

        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise LLMProviderError(f"Anthropic API call failed: {e}") from e

    def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        tool_choice: Optional[Dict] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        try:
            response = self.client.messages.create(**kwargs)

            text_parts = []
            tool_calls = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif getattr(block, "type", None) == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            return {
                "text": "\n".join(text_parts),
                "tool_calls": tool_calls,
                "stop_reason": getattr(response, "stop_reason", None),
            }
        except Exception as e:
            raise LLMProviderError(f"Anthropic tool call failed: {e}") from e


# ---------------------------------------------------------------------------
# QGenie Provider
# ---------------------------------------------------------------------------

class QGenieProvider(BaseLLMProvider):
    """
    QGenie model provider.
    Uses `qgenie.integrations.langchain.QGenieChat` for completions.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None

    @property
    def model_instance(self):
        if self._model is None:
            try:
                from qgenie.integrations.langchain import QGenieChat
            except ImportError:
                raise ProviderNotAvailableError(
                    "The 'qgenie' package is required for QGenie models. "
                    "Install it with: pip install qgenie"
                )
            self._model = QGenieChat(
                model=self.config.model,
                api_key=self.config.qgenie_api_key,
                timeout=self.config.timeout * 1000,  # QGenie uses milliseconds
            )
        return self._model

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            raise ProviderNotAvailableError(
                "langchain_core is required for QGenie provider. "
                "Install it with: pip install langchain-core"
            )

        lc_messages = []
        if system:
            lc_messages.append(SystemMessage(content=system))

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        try:
            start = time.monotonic()
            result = self.model_instance.invoke(
                lc_messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                repetition_penalty=1.1,
                top_k=50,
                top_p=0.95,
            )
            elapsed = time.monotonic() - start

            logger.debug(
                "QGenie API: model=%s elapsed=%.2fs",
                self.config.model, elapsed,
            )
            return result.content

        except Exception as e:
            logger.error("QGenie API error: %s", e)
            raise LLMProviderError(f"QGenie API call failed: {e}") from e


# ---------------------------------------------------------------------------
# VertexAI Provider (stub — extend when needed)
# ---------------------------------------------------------------------------

class VertexAIProvider(BaseLLMProvider):
    """
    Google Vertex AI provider (stub).
    Uses langchain_google_vertexai.ChatVertexAI for completions.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None

    @property
    def model_instance(self):
        if self._model is None:
            try:
                from langchain_google_vertexai import ChatVertexAI
            except ImportError:
                raise ProviderNotAvailableError(
                    "langchain-google-vertexai is required for Vertex AI models. "
                    "Install with: pip install langchain-google-vertexai"
                )
            self._model = ChatVertexAI(
                model_name=self.config.model,
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        return self._model

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            raise ProviderNotAvailableError("langchain-core is required.")

        lc_messages = []
        if system:
            lc_messages.append(SystemMessage(content=system))
        for msg in messages:
            lc_messages.append(HumanMessage(content=msg.get("content", "")))

        try:
            result = self.model_instance.invoke(lc_messages)
            return result.content
        except Exception as e:
            raise LLMProviderError(f"Vertex AI call failed: {e}") from e


# ---------------------------------------------------------------------------
# Azure OpenAI Provider (stub — extend when needed)
# ---------------------------------------------------------------------------

class AzureProvider(BaseLLMProvider):
    """
    Azure OpenAI provider (stub).
    Uses langchain_openai.AzureChatOpenAI for completions.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None

    @property
    def model_instance(self):
        if self._model is None:
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError:
                raise ProviderNotAvailableError(
                    "langchain-openai is required for Azure models. "
                    "Install with: pip install langchain-openai"
                )
            self._model = AzureChatOpenAI(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        return self._model

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            raise ProviderNotAvailableError("langchain-core is required.")

        lc_messages = []
        if system:
            lc_messages.append(SystemMessage(content=system))
        for msg in messages:
            lc_messages.append(HumanMessage(content=msg.get("content", "")))

        try:
            result = self.model_instance.invoke(lc_messages)
            return result.content
        except Exception as e:
            raise LLMProviderError(f"Azure OpenAI call failed: {e}") from e


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

# Registry: provider name -> provider class
PROVIDER_REGISTRY: Dict[str, type] = {
    "anthropic": AnthropicProvider,
    "qgenie": QGenieProvider,
    "vertexai": VertexAIProvider,
    "azure": AzureProvider,
}


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Factory function: create the correct provider based on config.

    Parses the provider from config.raw_model and instantiates the
    corresponding provider class.

    Args:
        config: LLMConfig with raw_model in "provider::model" format.

    Returns:
        An initialized BaseLLMProvider subclass.

    Raises:
        LLMProviderError: If provider is unknown.
    """
    provider_name = config.provider
    provider_cls = PROVIDER_REGISTRY.get(provider_name)

    if provider_cls is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise LLMProviderError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available providers: {available}. "
            f"Use format 'provider::model' (e.g., 'anthropic::claude-sonnet-4-20250514')."
        )

    logger.info("LLM provider: %s, model: %s", provider_name, config.model)
    return provider_cls(config)


# ═══════════════════════════════════════════════════════════════════════════
# Main LLMTools Class
# ═══════════════════════════════════════════════════════════════════════════

class LLMTools:
    """
    Generic multi-provider LLM toolkit for codebase analysis.

    Provider routing is automatic: set ``model`` to ``"anthropic::claude-sonnet-4-20250514"``
    for Claude, ``"qgenie::qwen2.5-14b-1m"`` for QGenie, etc.

    Features:
    - Multi-provider routing (Anthropic Claude, QGenie, VertexAI, Azure)
    - Structured intent extraction from natural language
    - JSON response parsing with fallback
    - Markdown prompt template rendering
    - Tool/function calling (Claude native tool_use)
    - Vector DB retrieval (optional, injected)
    - Pluggable prompt builder and keywords

    Usage:
        tools = LLMTools()                                    # uses config defaults
        tools = LLMTools(model="anthropic::claude-sonnet-4-20250514")  # Claude
        tools = LLMTools(model="qgenie::qwen2.5-14b-1m")              # QGenie

        response = tools.llm_call("Analyze this code...")
        intent   = tools.extract_intent_from_prompt("Compare A and B")
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        env_config=None,
        model: Optional[str] = None,
        vectordb=None,
        intent_prompt_builder: Optional[Callable[[str], str]] = None,
        full_report_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize LLM tools with automatic provider routing.

        Args:
            config: Explicit LLMConfig. If None, auto-loads from environment.
            env_config: Optional EnvConfig/GlobalConfig for config resolution.
            model: Override model string (e.g., "anthropic::claude-sonnet-4-20250514").
            vectordb: Optional VectorDB instance for retrieval methods.
            intent_prompt_builder: Pluggable function(user_input) -> system_prompt.
            full_report_keywords: Keywords for is_full_report_request().
        """
        # Build config
        if config:
            self.config = config
        else:
            self.config = LLMConfig.from_env(env_config)

        # Override model if explicitly provided
        if model:
            self.config.raw_model = model

        # Create the provider (auto-routed)
        self.provider = create_provider(self.config)

        # Optional vector DB
        self.vectordb = vectordb

        # Pluggable prompt builder
        self._intent_prompt_builder = intent_prompt_builder
        if self._intent_prompt_builder is None:
            try:
                from utils.prompts.prompts import get_intent_extraction_prompt
                self._intent_prompt_builder = get_intent_extraction_prompt
            except ImportError:
                self._intent_prompt_builder = self._default_intent_prompt

        # Full report keywords
        self._full_report_keywords = full_report_keywords
        if self._full_report_keywords is None:
            try:
                from utils.prompts.prompts import FULL_REPORT_KEYWORDS
                self._full_report_keywords = FULL_REPORT_KEYWORDS
            except ImportError:
                self._full_report_keywords = [
                    "all modules", "all", "trend", "all records",
                    "summary", "entire codebase",
                ]

        # Resolve prompt file path
        self.prompt_file_path = None
        if self.config.chat_prompt_file_path:
            try:
                self.prompt_file_path = self.resolve_relative_path(
                    self.config.chat_prompt_file_path
                )
            except Exception:
                pass

        self.logger = logger

    @classmethod
    def from_env(cls, env_config=None, **kwargs) -> "LLMTools":
        """Factory: build LLMTools from environment configuration."""
        return cls(config=LLMConfig.from_env(env_config), **kwargs)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_repo_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @classmethod
    def resolve_relative_path(cls, relative_path: Union[str, Path]) -> Path:
        if relative_path is None:
            raise ValueError("Path is not set.")
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return cls.get_repo_root() / p

    # ------------------------------------------------------------------
    # Core LLM Calls (provider-routed)
    # ------------------------------------------------------------------

    def llm_call(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Make a provider-routed LLM call and return the text response.

        If ``model`` is provided, temporarily switches to that provider
        for this single call.

        Args:
            prompt: The user message / prompt text.
            system: Optional system prompt.
            max_tokens: Override max tokens for this call.
            temperature: Override temperature for this call.
            model: Override model for this call (e.g., "qgenie::qwen2.5-14b-1m").

        Returns:
            The LLM's text response as a string.
        """
        # Use a different provider for this call if model is overridden
        provider = self.provider
        if model and model != self.config.raw_model:
            temp_config = LLMConfig(
                raw_model=model,
                anthropic_api_key=self.config.anthropic_api_key,
                qgenie_api_key=self.config.qgenie_api_key,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            provider = create_provider(temp_config)

        messages = [{"role": "user", "content": prompt}]
        return provider.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def llm_call_with_messages(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Multi-turn conversation call."""
        return self.provider.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def llm_call_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        tool_choice: Optional[Dict] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a tool-calling LLM request (native on Claude, fallback on others).

        Returns:
            {"text": str, "tool_calls": list, "stop_reason": str}
        """
        messages = [{"role": "user", "content": prompt}]
        return self.provider.complete_with_tools(
            messages=messages,
            tools=tools,
            system=system,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )

    def llm_call_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Force structured JSON output via tool_use (Claude) or JSON parsing (others).

        Args:
            prompt: User message.
            output_schema: JSON schema for the expected output.
            system: Optional system prompt.

        Returns:
            Parsed dict matching the schema.
        """
        # Try native tool-use structured output
        tool_def = {
            "name": "structured_output",
            "description": "Return the structured analysis result.",
            "input_schema": output_schema,
        }

        result = self.llm_call_with_tools(
            prompt=prompt,
            tools=[tool_def],
            system=system,
            tool_choice={"type": "tool", "name": "structured_output"},
        )

        for call in result.get("tool_calls", []):
            if call["name"] == "structured_output":
                return call["input"]

        # Fallback: parse JSON from text response
        if result.get("text"):
            json_str = self.extract_json_from_llm_response(result["text"])
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        raise LLMResponseError("No structured output returned from LLM.")

    # ------------------------------------------------------------------
    # Intent Extraction
    # ------------------------------------------------------------------

    def extract_intent_from_prompt(self, user_input_prompt: str) -> Dict[str, Any]:
        """
        Parse a natural language prompt into a structured intent object.

        Returns:
            Parsed intent dict with: intent, criteria, entities,
            fields_to_extract, output_format.
        """
        system_prompt = self._intent_prompt_builder(user_input_prompt)

        raw_response = self.llm_call(
            prompt=system_prompt,
            max_tokens=self.config.intent_max_tokens,
            temperature=self.config.intent_temperature,
        )

        json_str = self.extract_json_from_llm_response(raw_response)

        try:
            intent_obj = json.loads(json_str)
            if not isinstance(intent_obj, dict):
                raise ValueError(f"Expected dict, got {type(intent_obj)}")

            if "intent" not in intent_obj:
                intent_obj["intent"] = "retrieve"
            if intent_obj["intent"] not in ("retrieve", "compare", "aggregate"):
                intent_obj["intent"] = "retrieve"

            return intent_obj
        except json.JSONDecodeError as e:
            logger.error("Intent JSON parse failed: %s | raw: %s", e, raw_response[:500])
            raise IntentExtractionError(f"Failed to parse intent JSON: {e}") from e
        except Exception as e:
            logger.error("Intent extraction failed: %s", e)
            raise IntentExtractionError(f"Intent extraction failed: {e}") from e

    @staticmethod
    def _default_intent_prompt(user_input: str) -> str:
        """Minimal fallback intent prompt when prompts.py is unavailable."""
        return (
            "You are an expert codebase analysis assistant.\n\n"
            "Parse the user's query and return a JSON object with:\n"
            '- "intent": "retrieve" | "compare" | "aggregate"\n'
            '- "criteria": filter object (or {} for all)\n'
            '- "entities": list of entity descriptors (for compare)\n'
            '- "fields_to_extract": list of requested info types\n'
            '- "output_format": "summary" (default) | "table" | "list" | "json"\n\n'
            "Only return the JSON object.\n\n"
            f"User prompt: {user_input}\n"
        )

    # ------------------------------------------------------------------
    # Response Parsing Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def extract_json_from_llm_response(response: str) -> str:
        """
        Extract a JSON object from an LLM response that may be wrapped
        in markdown code fences or contain surrounding text.
        """
        if not isinstance(response, str):
            return str(response)

        response = response.strip()

        # Try markdown code fence
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response)
        if match:
            return match.group(1).strip()

        # Try to find a JSON object (first { to matching })
        brace_start = response.find("{")
        brace_end = response.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            return response[brace_start:brace_end + 1]

        # Try to find a JSON array
        bracket_start = response.find("[")
        bracket_end = response.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            return response[bracket_start:bracket_end + 1]

        return response

    @staticmethod
    def format_llm_response(agent_response: Any) -> str:
        """Normalize various LLM response formats into a plain string."""
        if agent_response is None:
            return "No response."

        if isinstance(agent_response, str):
            return agent_response

        # Handle objects with .content (Claude API response, LangChain message)
        if hasattr(agent_response, "content"):
            content = agent_response.content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                return "\n".join(parts) if parts else "No text content."
            return str(content)

        # Handle list of messages
        if isinstance(agent_response, list) and agent_response:
            for msg in reversed(agent_response):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "No content.")
                if hasattr(msg, "role") and msg.role == "assistant":
                    return getattr(msg, "content", "No content.")
            last = agent_response[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
            if hasattr(last, "content"):
                return str(last.content)
            return str(last)

        return str(agent_response)

    # ------------------------------------------------------------------
    # Prompt Template Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def update_markdown_prompt(
        md_filepath: Union[str, Path],
        input_str: str,
        records: Any,
        **extra_vars,
    ) -> Optional[str]:
        """Load a markdown prompt template and fill in variables."""
        try:
            path = Path(md_filepath)
            content = path.read_text(encoding="utf-8").strip()
            return content.format(
                records=records,
                input_str=input_str,
                **extra_vars,
            )
        except FileNotFoundError:
            logger.error("Prompt template not found: %s", md_filepath)
            return None
        except KeyError as e:
            logger.error("Missing template variable %s in %s", e, md_filepath)
            return None
        except Exception as e:
            logger.error("Error rendering prompt template %s: %s", md_filepath, e)
            return None

    # ------------------------------------------------------------------
    # Retrieval (optional — only active if vectordb is injected)
    # ------------------------------------------------------------------

    def retrieve_relevant_docs(self, input_str: str, top_k: int = 250) -> list:
        """
        Hybrid semantic/metadata retrieval using the injected VectorDB.

        Returns empty list if no vectordb is configured.
        """
        if self.vectordb is None:
            logger.warning("retrieve_relevant_docs called but no vectordb configured.")
            return []

        try:
            intent_obj = self.extract_intent_from_prompt(input_str)
        except Exception as e:
            logger.error("Intent extraction failed in retrieve: %s", e)
            intent_obj = {"intent": "retrieve", "criteria": {}}

        # Handle compare intent (multi-entity)
        if intent_obj.get("intent") == "compare" and "entities" in intent_obj:
            results = []
            for entity in intent_obj["entities"]:
                criteria = {k: v for k, v in entity.items() if k != "fields_to_extract"}
                query_str = " ".join(f"{k} {v}" for k, v in criteria.items())
                matched = self._semantic_and_metadata_search(query_str, criteria, top_k)
                results.append(matched)
            return results

        # Single-entity retrieve/aggregate
        criteria = intent_obj.get("criteria", {})
        return self._semantic_and_metadata_search(input_str, criteria, top_k)

    def _semantic_and_metadata_search(
        self, query_str: str, criteria: dict, top_k: int = 250,
    ) -> list:
        """Semantic search with metadata filtering."""
        docs = self.vectordb.retrieve(query_str, k=top_k)
        results = []
        for doc in docs:
            meta = getattr(doc, "metadata", None)
            if not meta:
                continue
            matches = True
            for k, v in criteria.items():
                meta_val = meta.get(k) if isinstance(meta, dict) else getattr(meta, k, None)
                data_val = getattr(doc, "data", {}).get(k) if hasattr(doc, "data") else None
                if meta_val == v or data_val == v:
                    continue
                matches = False
                break
            if matches:
                results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def is_full_report_request(self, user_input: str) -> bool:
        """Check if the user is requesting a full/all-modules report."""
        lower = user_input.lower()
        return any(kw in lower for kw in self._full_report_keywords)

    @staticmethod
    def is_uuid(value: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def count_tokens_approx(text: str) -> int:
        """Approximate token count (~4 chars/token heuristic)."""
        return len(text) // 4

    def truncate_to_token_budget(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to fit within a token budget."""
        budget = max_tokens or self.config.max_prompt_tokens
        current = self.count_tokens_approx(text)
        if current <= budget:
            return text
        char_limit = budget * 4
        return text[:char_limit] + "\n\n[... truncated due to token limit ...]"

    def metadata_filtering(self, all_records: list, **criteria) -> list:
        """Filter records matching all given metadata/data criteria."""
        def match(rec):
            for key, val in criteria.items():
                meta_val = rec.get("metadata", {}).get(key)
                data_val = rec.get("data", {}).get(key)
                if meta_val == val or data_val == val:
                    continue
                return False
            return True
        return [rec for rec in all_records if match(rec)]

    # ------------------------------------------------------------------
    # Tool Registry
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return Claude-compatible tool definitions for this toolkit."""
        return [
            {
                "name": "analyze_code",
                "description": "Analyze code for quality, security, complexity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Source code to analyze."},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["quality", "security", "complexity", "dependencies", "architecture", "full"],
                        },
                        "language": {"type": "string", "description": "Programming language."},
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "extract_intent",
                "description": "Extract structured intent from a natural language query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query."},
                    },
                    "required": ["query"],
                },
            },
        ]

    def get_tools_map(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build a tool map for agent/function-calling integration.

        Backward-compatible with the old LangChain-based get_tools_map.
        """
        tool_mapping = {
            "retrieve_relevant_docs": self.retrieve_relevant_docs,
            "update_markdown_prompt": self.update_markdown_prompt,
            "llm_call": self.llm_call,
            "format_llm_response": self.format_llm_response,
        }

        if names is None:
            names = list(tool_mapping.keys())

        tools_map = {}
        all_tools = []
        for name in names:
            if name not in tool_mapping:
                logger.warning("Unknown tool: %s", name)
                continue
            tool = tool_mapping[name]
            tools_map[name] = {"call": tool}
            all_tools.append(tool)

        tools_map["__all__"] = {"call": all_tools}
        return tools_map

    def get_all_available_tools(self) -> List[str]:
        """Return names of all available tool methods."""
        return [
            "llm_call", "llm_call_with_messages", "llm_call_with_tools",
            "llm_call_structured", "extract_intent_from_prompt",
            "retrieve_relevant_docs", "update_markdown_prompt",
            "format_llm_response", "truncate_to_token_budget",
        ]

    # ------------------------------------------------------------------
    # Provider Info
    # ------------------------------------------------------------------

    def get_provider_info(self) -> Dict[str, str]:
        """Return current provider and model information."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "raw_model": self.config.raw_model,
            "provider_class": type(self.provider).__name__,
        }

    def switch_model(self, model: str) -> None:
        """
        Switch the active model/provider at runtime.

        Args:
            model: New model string (e.g., "qgenie::qwen2.5-14b-1m").
        """
        self.config.raw_model = model
        self.provider = create_provider(self.config)
        logger.info("Switched to: %s (%s)", self.config.provider, self.config.model)

    def __repr__(self) -> str:
        return (
            f"LLMTools(provider='{self.config.provider}', "
            f"model='{self.config.model}')"
        )
