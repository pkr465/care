"""
Enhanced LLM integrations for C/C++ codebase analysis (QGenie-only)

This module provides all LLM-related functionality for the C/C++ analysis workflow,
with a focus on:
- Code health metrics (security, quality, complexity, maintainability, docs, tests)
- Dependency and architecture analysis
- Professional, intent-aware responses for reports and summaries

Compatible with existing LLMTools and QGenie integration.
"""

import json
import logging
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import HumanMessage

from utils.parsers.env_parser import EnvConfig
from .prompt_templates import PromptTemplates

# QGenie imports
try:
    from qgenie.integrations.langchain import QGenieChat
    QGENIE_AVAILABLE = True
except ImportError:
    QGENIE_AVAILABLE = False

import logging
import sys
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # <-- add this to send logs to stdout
    force=True          # <-- ensure configuration is enforced, even if already set
)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers (restricted to QGenie + Mock for this integration)."""
    QGENIE = "qgenie"
    MOCK = "mock"


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    cost_estimate: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMConfig:
    """LLM configuration (QGenie-focused)."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 15000  # Increased for C/C++ analysis
    temperature: float = 0.1  # Lower for more consistent analysis
    timeout: int = 15000      # ms; aligned with existing usage
    max_retries: int = 3
    retry_delay: float = 1.0

    # QGenie-specific settings
    repetition_penalty: float = 1.1
    top_k: int = 50
    top_p: float = 0.95

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Cost tracking
    track_costs: bool = True


class LLMIntegrations:
    """
    QGenie-based LLM integrations for C/C++ codebase analysis.

    - Provides LLMTools-compatible APIs (llm_call, extract_json_from_llm_response,
      format_llm_response, get_tools_map, get_all_available_tools)
    - Adds intent extraction tailored to code analysis metrics and architecture queries.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        env_config: Optional[EnvConfig] = None,
    ):
        self.env_config = env_config or EnvConfig()
        self.config = config or self._create_config_from_env()
        self.logger = logging.getLogger(__name__)
        self.prompt_templates = PromptTemplates()

        # QGenie client or Mock client
        self.client = None

        # Rate limiting & caching
        self._request_times: List[float] = []
        self._token_usage: List[tuple[int, float]] = []
        self._cache: Dict[str, Any] = {}

        # Cost tracking
        self._total_cost: float = 0.0
        self._total_tokens: int = 0

        self._initialize_client()

    # ------------------------------------------------------------------ #
    # Configuration & client initialization
    # ------------------------------------------------------------------ #

    def _create_config_from_env(self) -> LLMConfig:
        """Create QGenie-oriented config from environment variables."""
        provider = LLMProvider.QGENIE if QGENIE_AVAILABLE else LLMProvider.MOCK

        return LLMConfig(
            provider=provider,
            model=self.env_config.get("LLM_MODEL", "qgenie-default-model"),
            api_key=self.env_config.get("QGENIE_API_KEY"),
            max_tokens=int(self.env_config.get("LLM_MAX_TOKENS", "15000")),
            temperature=float(self.env_config.get("LLM_TEMPERATURE", "0.1")),
            timeout=int(self.env_config.get("LLM_TIMEOUT", "15000")),
            repetition_penalty=float(self.env_config.get("LLM_REPETITION_PENALTY", "1.1")),
            top_k=int(self.env_config.get("LLM_TOP_K", "50")),
            top_p=float(self.env_config.get("LLM_TOP_P", "0.95")),
        )

    def _initialize_client(self):
        """Initialize QGenie or Mock client."""
        try:
            if self.config.provider == LLMProvider.QGENIE:
                self._initialize_qgenie_client()
            else:
                self._initialize_mock_client()
        except Exception as e:
            self.logger.warning(f"Failed to initialize QGenie client: {e}")
            self.logger.info("Falling back to mock client")
            self._initialize_mock_client()

    def _initialize_qgenie_client(self):
        """Initialize QGenie client (aligned with LLMTools)."""
        if not QGENIE_AVAILABLE:
            raise ImportError("QGenie integration not available")

        if not self.config.api_key:
            self.config.api_key = self.env_config.get("QGENIE_API_KEY")

        if not self.config.api_key:
            raise ValueError("QGenie API key not found in configuration")

        self.client = QGenieChat(
            model=self.config.model,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        self.logger.info("QGenie client initialized successfully")

    def _initialize_mock_client(self):
        """Initialize mock client for testing."""
        self.client = MockLLMClient()
        self.logger.info("Mock LLM client initialized (QGenie not available)")

    # ------------------------------------------------------------------ #
    # Caching, rate limiting, and usage tracking
    # ------------------------------------------------------------------ #

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        cache_data = {
            "prompt": prompt,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[LLMResponse]:
        if not self.config.enable_cache:
            return None
        if cache_key in self._cache:
            cached_response, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                self.logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached_response
            else:
                del self._cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: LLMResponse):
        if self.config.enable_cache:
            self._cache[cache_key] = (response, time.time())

    def _check_rate_limits(self):
        now = time.time()
        # Clean old entries
        self._request_times = [t for t in self._request_times if now - t < 60]
        self._token_usage = [(tokens, t) for tokens, t in self._token_usage if now - t < 60]

        # Request limit
        if len(self._request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Request rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        # Token limit
        total_tokens = sum(tokens for tokens, _ in self._token_usage)
        if total_tokens >= self.config.tokens_per_minute:
            oldest_time = min(t for _, t in self._token_usage)
            sleep_time = 60 - (now - oldest_time)
            if sleep_time > 0:
                self.logger.info(f"Token rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

    def _estimate_cost(self, tokens_used: int) -> float:
        if not self.config.track_costs:
            return 0.0
        # Simple flat estimate for QGenie
        rate_per_1k = 0.001
        return (tokens_used / 1000.0) * rate_per_1k

    def _update_usage_stats(self, tokens_used: int, cost: float):
        now = time.time()
        self._request_times.append(now)
        self._token_usage.append((tokens_used, now))
        self._total_tokens += tokens_used
        self._total_cost += cost

    # ------------------------------------------------------------------ #
    # Core query + LLMTools-compatible llm_call
    # ------------------------------------------------------------------ #

    def llm_call(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        LLMTools-compatible call.

        Args:
            prompt: Prompt text
            model: Optional model override (ignored if using fixed config)
            **kwargs: Additional optional parameters

        Returns:
            Response content as string
        """
        try:
            response = self.query(prompt, **kwargs)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            return f"LLM invocation failed: {e}"

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Query the LLM with the given prompt, with caching and retries.

        Returns:
            LLMResponse with content and metadata.
        """
        cache_key = self._get_cache_key(prompt, **kwargs)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        self._check_rate_limits()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                start = time.time()
                response = self._query_qgenie(prompt, **kwargs)
                elapsed = time.time() - start

                response.response_time = elapsed
                response.cost_estimate = self._estimate_cost(response.tokens_used)
                self._update_usage_stats(response.tokens_used, response.cost_estimate)
                self._cache_response(cache_key, response)

                return response
            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2**attempt))

        raise Exception(f"All {self.config.max_retries} attempts failed. Last error: {last_error}")

    def _query_qgenie(self, prompt: str, **kwargs) -> LLMResponse:
        """Low-level QGenie query, compatible with LLMTools behavior."""
        if isinstance(self.client, MockLLMClient):
            return self.client.generate_response(prompt)

        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        messages = [HumanMessage(content=prompt)]
        result = self.client.invoke(
            messages,
            max_tokens=max_tokens,
            repetition_penalty=self.config.repetition_penalty,
            temperature=temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

        tokens_used = len(prompt.split()) + len(result.content.split())

        return LLMResponse(
            content=result.content,
            provider=self.config.provider.value,
            model=self.config.model,
            tokens_used=tokens_used,
            metadata={
                "repetition_penalty": self.config.repetition_penalty,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            },
        )

    # ------------------------------------------------------------------ #
    # Intent extraction (metrics & code analysis focused)
    # ------------------------------------------------------------------ #

    def extract_intent_from_prompt(self, user_input_prompt):
        """
        Uses an LLM to parse the user prompt and return a structured intent object (dict).
        Supports 'retrieve', 'compare', and 'aggregate' style queries.
        Handles 'all test runs' (no specific criterion) and 'no criterion' cases explicitly.
        """
        system_prompt = (
            "You are an expert codebase analysis assistant.\n\n"
            "Your task:\n"
            "Given a natural-language user prompt about a codebase, analyze it and return a JSON object that captures "
            "the user's intent for codebase/module/dependency/architecture queries.\n\n"
            "You MUST:\n"
            "- Only output a JSON object (no extra text).\n"
            "- Infer the most likely intent and fill as many fields as possible, even if the user is not precise.\n"
            "- Stay within the supported schema and intents listed below.\n\n"
            "--------------------------------\n"
            "Supported intents\n"
            "--------------------------------\n"
            'The "intent" field must be one of:\n'
            '- \"retrieve\"   – Get information about specific entities (modules, files, components, branches, services, tests, etc.).\n"
            '- \"compare\"    – Compare two or more entities using specific criteria or metrics.\n'
            '- \"aggregate\"  – Aggregate or summarize information across many entities (e.g., whole codebase, all modules, all services).\n\n'
            "--------------------------------\n"
            "Core JSON schema\n"
            "--------------------------------\n"
            "You must return a single JSON object with some or all of the following fields:\n\n"
            '- intent: \"retrieve\" | \"compare\" | \"aggregate\"\n\n'
            "- criteria: object (for \"retrieve\" and \"aggregate\")\n"
            "  - A filter describing which parts of the codebase the user is interested in.\n"
            "  - Examples:\n"
            "    - {\"module\": \"module_A\"}\n"
            "    - {\"file_path\": \"src/core/utils.py\"}\n"
            "    - {\"service\": \"auth_service\"}\n"
            "    - {\"branch\": \"feature/login_v2\"}\n"
            "    - {\"repository\": \"my-repo\"}\n"
            "    - {\"layer\": \"data_access\"}\n"
            "    - {\"tag\": \"payment\"}\n"
            "  - If the user does not specify a particular subset (e.g. \"all modules\", \"entire codebase\", \"global metrics\"), "
            "use an empty object: {}.\n\n"
            "- entities: array (for \"compare\")\n"
            "  - List of entities to compare.\n"
            "  - Each entity is an object that may include identifiers like:\n"
            "    - {\"module\": \"module_A\"}\n"
            "    - {\"module\": \"module_B\"}\n"
            "    - {\"branch\": \"feature_x\"}\n"
            "    - {\"branch\": \"main\"}\n"
            "    - {\"file_path\": \"src/a.py\"}\n"
            "    - {\"service\": \"checkout\"}\n"
            "  - Use as many fields as needed to disambiguate (e.g., include both \"repository\" and \"branch\" if mentioned).\n\n"
            "- fields_to_extract: array of strings\n"
            "  - The concrete types of information the user is asking for.\n"
            "  - This can include:\n"
            "    - Structural / architecture:\n"
            "      - \"modules\"\n"
            "      - \"components\"\n"
            "      - \"services\"\n"
            "      - \"dependencies\"\n"
            "      - \"call_graph\"\n"
            "      - \"architecture_overview\"\n"
            "      - \"interfaces\"\n"
            "      - \"data_flows\"\n"
            "      - \"entry_points\"\n"
            "    - Code / docs:\n"
            "      - \"code\"\n"
            "      - \"code_snippets\"\n"
            "      - \"documentation\"\n"
            "      - \"inline_comments\"\n"
            "      - \"api_docs\"\n"
            "      - \"design_docs\"\n"
            "    - Tests:\n"
            "      - \"tests\"\n"
            "      - \"unit_tests\"\n"
            "      - \"integration_tests\"\n"
            "      - \"test_coverage\"\n"
            "      - \"test_plan\"\n"
            "      - \"test_recommendations\"\n"
            "    - Health & quality metrics (explicitly support these):\n"
            "      - \"complexity\"\n"
            "      - \"dependency_metrics\"\n"
            "      - \"documentation_coverage\"\n"
            "      - \"maintainability\"\n"
            "      - \"code_smells\"\n"
            "      - \"quality\"\n"
            "      - \"security_issues\"\n"
            "      - \"security_risks\"\n"
            "      - \"vulnerabilities\"\n"
            "      - \"testability\"\n"
            "      - \"technical_debt\"\n"
            "      - \"style_compliance\"\n"
            "      - \"lint_issues\"\n"
            "    - Plans / recommendations:\n"
            "      - \"modularization_plan\"\n"
            "      - \"refactoring_suggestions\"\n"
            "      - \"architecture_recommendations\"\n"
            "      - \"performance_recommendations\"\n"
            "      - \"security_recommendations\"\n"
            "      - \"test_strategy\"\n"
            "    - Versioning / branching:\n"
            "      - \"diff\"\n"
            "      - \"changes\"\n"
            "      - \"changelog\"\n"
            "      - \"merge_risks\"\n"
            "      - \"breaking_changes\"\n\n"
            "- output_format: string\n"
            "  - The preferred output shape if the user specifies one. Examples:\n"
            "    - \"table\"              // conceptual tabular representation\n"
            "    - \"tabular_list\"       // table represented as a list of rows / list of objects\n"
            "    - \"array\"\n"
            "    - \"list\"\n"
            "    - \"summary\"\n"
            "    - \"detailed_summary\"\n"
            "    - \"graph\"\n"
            "    - \"json\"\n"
            "    - \"code_block\"\n"
            "    - \"markdown\"\n"
            "  - If the user says \"as a list of rows\", \"table as a list\", \"tabular output as a list\", or similar,\n"
            "    set: \"output_format\": \"tabular_list\".\n"
            "  - If the user does not specify, default to \"summary\".\n\n"
            "- additional_context: object (optional)\n"
            "  - Any extra intent-related details that help downstream processing but are not simple filters.\n"
            "  - Examples:\n"
            "    - {\"time_range\": \"last_30_days\"}\n"
            "    - {\"include_private_apis\": true}\n"
            "    - {\"focus_on\": [\"performance\", \"security\"]}\n"
            "    - {\"thresholds\": {\"complexity\": \"high\", \"coverage\": \"< 80%\"}}\n"
            "    - {\"view\": \"manager\"}\n"
            "    - {\"view\": \"developer\"}\n\n"
            "--------------------------------\n"
            "Interpreting user requests\n"
            "--------------------------------\n"
            "1. Retrieval (\"retrieve\"):\n"
            "   - Use when the user asks for information about specific entities or a narrowly defined subset.\n"
            "   - Examples:\n"
            "     - \"Provide modularization plan and code for module_A\"\n"
            "       -> intent: \"retrieve\"\n"
            "       -> criteria: {\"module\": \"module_A\"}\n"
            "       -> fields_to_extract: [\"modularization_plan\", \"code\"]\n"
            "       -> output_format: \"summary\"\n"
            "     - \"Show the dependencies and complexity for auth_service\"\n"
            "       -> intent: \"retrieve\"\n"
            "       -> criteria: {\"service\": \"auth_service\"}\n"
            "       -> fields_to_extract: [\"dependencies\", \"complexity\"]\n"
            "       -> output_format: \"summary\"\n"
            "     - \"What tests cover file src/core/utils.py?\"\n"
            "       -> intent: \"retrieve\"\n"
            "       -> criteria: {\"file_path\": \"src/core/utils.py\"}\n"
            "       -> fields_to_extract: [\"tests\", \"test_coverage\"]\n"
            "       -> output_format: \"list\"\n\n"
            "2. Comparison (\"compare\"):\n"
            "   - Use when the user explicitly wants differences, comparison, or trade-offs between multiple entities.\n"
            "   - List each entity in \"entities\".\n"
            "   - Put requested comparison dimensions in \"fields_to_extract\".\n"
            "   - Examples:\n"
            "     - \"Compare the dependencies for module A and B in a tabular format.\"\n"
            "       -> {\n"
            "            \"intent\": \"compare\",\n"
            "            \"entities\": [\n"
            "              {\"module\": \"module_A\"},\n"
            "              {\"module\": \"module_B\"}\n"
            "            ],\n"
            "            \"fields_to_extract\": [\"dependencies\"],\n"
            "            \"output_format\": \"table\"\n"
            "          }\n"
            "     - \"Compare code quality, complexity, and test coverage between branch main and feature/login_v2\"\n"
            "       -> {\n"
            "            \"intent\": \"compare\",\n"
            "            \"entities\": [\n"
            "              {\"branch\": \"main\"},\n"
            "              {\"branch\": \"feature/login_v2\"}\n"
            "            ],\n"
            "            \"fields_to_extract\": [\n"
            "              \"quality\",\n"
            "              \"complexity\",\n"
            "              \"test_coverage\"\n"
            "            ],\n"
            "            \"output_format\": \"summary\"\n"
            "          }\n"
            "     - \"Compare checkout and payments modules with their key metrics (complexity, quality, test coverage) as a list of rows\"\n"
            "       -> {\n"
            "            \"intent\": \"compare\",\n"
            "            \"entities\": [\n"
            "              {\"module\": \"checkout\"},\n"
            "              {\"module\": \"payments\"}\n"
            "            ],\n"
            "            \"fields_to_extract\": [\n"
            "              \"complexity\",\n"
            "              \"quality\",\n"
            "              \"test_coverage\"\n"
            "            ],\n"
            "            \"output_format\": \"tabular_list\"\n"
            "          }\n\n"
            "3. Aggregation (\"aggregate\"):\n"
            "   - Use when the user asks for summaries, roll-ups, or overviews across many entities or the entire codebase.\n"
            "   - If the scope is \"all modules\" or \"entire codebase\", use empty criteria: {}.\n"
            "   - Examples:\n"
            "     - \"Show all modules and their dependencies as a table.\"\n"
            "       -> {\n"
            "            \"intent\": \"aggregate\",\n"
            "            \"criteria\": {},\n"
            "            \"fields_to_extract\": [\"module\", \"dependencies\"],\n"
            "            \"output_format\": \"table\"\n"
            "          }\n"
            "     - \"Summarize the overall health of the codebase including complexity, test coverage, and security issues.\"\n"
            "       -> {\n"
            "            \"intent\": \"aggregate\",\n"
            "            \"criteria\": {},\n"
            "            \"fields_to_extract\": [\n"
            "              \"complexity\",\n"
            "              \"test_coverage\",\n"
            "              \"security_issues\",\n"
            "              \"quality\",\n"
            "              \"maintainability\"\n"
            "            ],\n"
            "            \"output_format\": \"summary\"\n"
            "          }\n"
            "     - \"List each service with its documentation coverage and testability as a list of records\"\n"
            "       -> {\n"
            "            \"intent\": \"aggregate\",\n"
            "            \"criteria\": {},\n"
            "            \"fields_to_extract\": [\n"
            "              \"service\",\n"
            "              \"documentation_coverage\",\n"
            "              \"testability\"\n"
            "            ],\n"
            "            \"output_format\": \"tabular_list\"\n"
            "          }\n\n"
            "--------------------------------\n"
            "Handling code / documentation / recommendations\n"
            "--------------------------------\n"
            "- If the user asks for:\n"
            "  - Code examples or snippets → include \"code\" or \"code_snippets\".\n"
            "  - Documentation or design explanations → include \"documentation\", \"design_docs\", or \"architecture_overview\".\n"
            "  - Architecture or refactoring advice → include \"architecture_recommendations\", \"modularization_plan\", or \"refactoring_suggestions\".\n"
            "  - Health, quality, or metrics → map to one or more of:\n"
            "    - \"complexity\"\n"
            "    - \"dependency_metrics\"\n"
            "    - \"documentation_coverage\"\n"
            "    - \"maintainability\"\n"
            "    - \"quality\"\n"
            "    - \"security_issues\"\n"
            "    - \"testability\"\n"
            "    - \"technical_debt\"\n"
            "    - \"code_smells\"\n"
            "    - \"test_coverage\"\n\n"
            "--------------------------------\n"
            "Ambiguous or underspecified queries\n"
            "--------------------------------\n"
            "- If the user does not provide a specific criterion (e.g., no particular module, file, or service):\n"
            "  - Treat the query as applying to the entire codebase or all modules.\n"
            "  - Set \"criteria\": {}.\n"
            "- If the user asks something like:\n"
            "  - \"Give me a summary of the codebase architecture and its main dependencies\"\n"
            "    - Use:\n"
            "      - intent: \"aggregate\"\n"
            "      - criteria: {}\n"
            "      - fields_to_extract: [\"architecture_overview\", \"dependencies\"]\n"
            "      - output_format: \"summary\"\n"
            "- If the user hints at health metrics without naming them exactly:\n"
            "  - Map phrases to relevant fields, e.g.:\n"
            "    - \"code health\" → [\"complexity\", \"maintainability\", \"quality\", \"test_coverage\"]\n"
            "    - \"is this module safe/secure?\" → [\"security_issues\", \"vulnerabilities\"]\n"
            "    - \"how easy is this to change?\" → [\"maintainability\", \"testability\", \"technical_debt\"]\n\n"
            "--------------------------------\n"
            "Final requirement\n"
            "--------------------------------\n"
            "Only return the JSON object with the inferred structure and fields.\n\n"
            f"User prompt: {user_input_prompt}\n"
        )
        logger.debug(f"[extract_intent_from_prompt] system_prompt: {system_prompt}")

        raw_llm_response = self.llm_call(system_prompt)
        llm_response = self.extract_json_from_llm_response(raw_llm_response)
        logger.debug(f"[extract_intent_from_prompt] LLM response: {llm_response}")

        try:
            if not isinstance(llm_response, str):
                raise ValueError(f"Expected string from llm_call, got {type(llm_response)}")
            intent_obj = json.loads(llm_response)
            if not isinstance(intent_obj, dict):
                raise ValueError("Parsed LLM response is not a dict.")
            return intent_obj
        except Exception as e:
            logger.error(
                f"[extract_intent_from_prompt] Failed to parse LLM response. "
                f"Prompt: {user_input_prompt} | Response: {llm_response} | Error: {e}"
            )
            raise ValueError("Intent extraction failed") from e

    # ------------------------------------------------------------------ #
    # Response helpers (LLMTools compatibility)
    # ------------------------------------------------------------------ #

    def extract_json_from_llm_response(self, response: str) -> str:
        """
        Extract JSON object from an LLM response that may be wrapped
        in a Markdown code block (```json ... ```).
        """
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response)
        if match:
            return match.group(1).strip()
        return response.strip()

    def format_llm_response(self, agent_response) -> str:
        """
        Format LLM/agent response into a plain string (LLMTools-compatible).
        """
        try:
            if isinstance(agent_response, LLMResponse):
                return agent_response.content

            if isinstance(agent_response, list) and agent_response:
                assistant_msgs = []
                for msg in agent_response:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        assistant_msgs.append(msg)
                    elif hasattr(msg, "role") and msg.role == "assistant":
                        assistant_msgs.append(msg)

                if assistant_msgs:
                    last = assistant_msgs[-1]
                    if isinstance(last, dict):
                        return last.get("content", "No response.")
                    if hasattr(last, "content"):
                        return last.content
                    return str(last)

                last = agent_response[-1]
                if isinstance(last, dict):
                    return last.get("content", str(last))
                if hasattr(last, "content"):
                    return last.content
                return str(last)

            if hasattr(agent_response, "role") and hasattr(agent_response, "content"):
                return agent_response.content

            if isinstance(agent_response, str):
                return agent_response

            if agent_response is None:
                return "No response."

            return f"Unknown response type: {type(agent_response)}"
        except Exception as e:
            return f"Error extracting answer: {e}"

    # ------------------------------------------------------------------ #
    # High-level analysis methods (QGenie-backed, using PromptTemplates)
    # ------------------------------------------------------------------ #

    def get_codebase_insights(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get high-level LLM insights about the overall C/C++ codebase."""
        try:
            prompt = self.prompt_templates.get_codebase_insights_prompt(file_cache)
            response = self.query(prompt)

            return {
                "architecture_assessment": self._extract_section(response.content, "Architecture Assessment"),
                "technology_stack": self._extract_section(response.content, "Technology Stack"),
                "project_type": self._extract_section(response.content, "Project Type"),
                "development_practices": self._extract_section(response.content, "Development Practices"),
                "potential_concerns": self._extract_section(response.content, "Potential Concerns"),
                "recommendations": self._extract_section(response.content, "Recommendations"),
                "raw_response": response.content,
                "metadata": {
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "response_time": response.response_time,
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting codebase insights: {e}")
            return {"error": str(e)}

    def analyze_dependencies(self, dependency_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM analysis of dependency structure and architecture implications."""
        try:
            prompt = self.prompt_templates.get_dependency_analysis_prompt(dependency_graph)
            response = self.query(prompt)

            return {
                "dependency_health": self._extract_section(response.content, "Dependency Health"),
                "coupling_analysis": self._extract_section(response.content, "Coupling Analysis"),
                "circular_dependencies": self._extract_section(response.content, "Circular Dependencies"),
                "architecture_implications": self._extract_section(response.content, "Architecture Implications"),
                "refactoring_opportunities": self._extract_section(response.content, "Refactoring Opportunities"),
                "cpp_best_practices": self._extract_section(response.content, "C/C++ Best Practices"),
                "build_system_impact": self._extract_section(response.content, "Build System Impact"),
                "raw_response": response.content,
                "metadata": {
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "response_time": response.response_time,
                },
            }
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {e}")
            return {"error": str(e)}

    def analyze_health_metrics(self, health_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM commentary and recommendations based on computed health metrics."""
        try:
            prompt = self.prompt_templates.get_health_metrics_analysis_prompt(health_metrics)
            response = self.query(prompt)

            return {
                "critical_issues": self._extract_section(response.content, "Critical Issues"),
                "security_concerns": self._extract_section(response.content, "Security Concerns"),
                "quality_improvements": self._extract_section(response.content, "Quality Improvements"),
                "complexity_management": self._extract_section(response.content, "Complexity Management"),
                "maintainability_strategy": self._extract_section(response.content, "Maintainability Strategy"),
                "documentation_gaps": self._extract_section(response.content, "Documentation Gaps"),
                "testing_strategy": self._extract_section(response.content, "Testing Strategy"),
                "priority_roadmap": self._extract_section(response.content, "Priority Roadmap"),
                "raw_response": response.content,
                "metadata": {
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "response_time": response.response_time,
                },
            }
        except Exception as e:
            self.logger.error(f"Error analyzing health metrics: {e}")
            return {"error": str(e)}

    def get_security_analysis(self, security_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get a focused security analysis from the LLM."""
        try:
            prompt = self.prompt_templates.get_security_focus_prompt(security_metrics)
            response = self.query(prompt)

            return {
                "memory_safety": self._extract_section(response.content, "Memory Safety"),
                "buffer_overflow_prevention": self._extract_section(response.content, "Buffer Overflow Prevention"),
                "input_validation": self._extract_section(response.content, "Input Validation"),
                "secure_coding_practices": self._extract_section(response.content, "Secure Coding Practices"),
                "static_analysis": self._extract_section(response.content, "Static Analysis"),
                "runtime_protection": self._extract_section(response.content, "Runtime Protection"),
                "code_review_focus": self._extract_section(response.content, "Code Review Focus"),
                "vulnerability_remediation": self._extract_section(response.content, "Vulnerability Remediation"),
                "raw_response": response.content,
                "metadata": {
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "response_time": response.response_time,
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting security analysis: {e}")
            return {"error": str(e)}

    def generate_final_report(
        self,
        summary: Dict[str, Any],
        dependency_graph: Dict[str, Any],
        documentation: Dict[str, Any],
        modularization_plan: Dict[str, Any],
        validation_report: Dict[str, Any],
        health_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive final report with professional, actionable recommendations.
        """
        try:
            prompt = self.prompt_templates.get_final_report_prompt(
                summary,
                dependency_graph,
                documentation,
                modularization_plan,
                validation_report,
                health_metrics,
            )
            response = self.query(prompt, max_tokens=3000)

            return {
                "executive_summary": self._extract_section(response.content, "Executive Summary"),
                "critical_issues": self._extract_section(response.content, "Critical Issues"),
                "strategic_recommendations": self._extract_section(response.content, "Strategic Recommendations"),
                "implementation_roadmap": self._extract_section(response.content, "Implementation Roadmap"),
                "resource_requirements": self._extract_section(response.content, "Resource Requirements"),
                "risk_assessment": self._extract_section(response.content, "Risk Assessment"),
                "success_metrics": self._extract_section(response.content, "Success Metrics"),
                "technology_modernization": self._extract_section(response.content, "Technology Modernization"),
                "team_development": self._extract_section(response.content, "Team Development"),
                "monitoring_strategy": self._extract_section(response.content, "Monitoring Strategy"),
                "raw_response": response.content,
                "metadata": {
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "response_time": response.response_time,
                },
            }
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            return {"error": str(e)}

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract a named section from a markdown-style response."""
        try:
            patterns = [
                f"**{section_name}:**",
                f"## {section_name}",
                f"{section_name}:",
                f"**{section_name.upper()}:**",
                f"### {section_name}",
            ]

            for pattern in patterns:
                if pattern in response:
                    start_idx = response.find(pattern)
                    if start_idx == -1:
                        continue
                    content_start = start_idx + len(pattern)
                    content = response[content_start:]

                    next_section_patterns = [
                        r"\n\*\*[^*]+\*\*:",
                        r"\n## [^\n]+",
                        r"\n### [^\n]+",
                        r"\n\d+\. \*\*[^*]+\*\*:",
                    ]
                    end_idx = len(content)
                    for next_pattern in next_section_patterns:
                        m = re.search(next_pattern, content)
                        if m:
                            end_idx = min(end_idx, m.start())
                    return content[:end_idx].strip()

            # Fallback: return a trimmed prefix
            return response[:500] + "..." if len(response) > 500 else response
        except Exception as e:
            self.logger.error(f"Error extracting section '{section_name}': {e}")
            return f"Error extracting section: {e}"

    # ------------------------------------------------------------------ #
    # Utilities: stats, cache, tools map
    # ------------------------------------------------------------------ #

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": len(self._request_times),
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "cache_size": len(self._cache),
            "provider": self.config.provider.value,
            "model": self.config.model,
        }

    def clear_cache(self):
        self._cache.clear()
        self.logger.info("LLM response cache cleared")

    def save_cache(self, file_path: str):
        try:
            data = {
                "cache": self._cache,
                "config": {
                    "provider": self.config.provider.value,
                    "model": self.config.model,
                    "cache_ttl": self.config.cache_ttl,
                },
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"LLM cache saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def load_cache(self, file_path: str):
        try:
            if not Path(file_path).exists():
                return
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            if (
                cfg.get("provider") == self.config.provider.value
                and cfg.get("model") == self.config.model
            ):
                self._cache = data.get("cache", {})
                self.logger.info(f"LLM cache loaded from {file_path}")
            else:
                self.logger.warning("Cache incompatible with current configuration")
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")

    def get_tools_map(self, names: List[str]) -> Dict[str, Any]:
        """
        LLMTools-compatible tool map.

        Exposes only QGenie-backed tools and helpers.
        """
        tool_mapping = {
            "llm_call": self.llm_call,
            "extract_json_from_llm_response": self.extract_json_from_llm_response,
            "format_llm_response": self.format_llm_response,
            "get_codebase_insights": self.get_codebase_insights,
            "analyze_dependencies": self.analyze_dependencies,
            "analyze_health_metrics": self.analyze_health_metrics,
            "get_security_analysis": self.get_security_analysis,
            "generate_final_report": self.generate_final_report,
            "extract_intent_from_prompt": self.extract_intent_from_prompt,
        }

        tools_map: Dict[str, Any] = {}
        for name in names:
            if name in tool_mapping:
                tool = tool_mapping[name]
                tools_map[name] = {"call": tool}
                try:
                    tools_map[name]["schema"] = convert_to_openai_tool(tool)
                except Exception as e:
                    self.logger.warning(f"Could not create schema for tool {name}: {e}")
                    tools_map[name]["schema"] = None

        # Convenience: pack all calls/specs under "__all__"
        all_calls = [v["call"] for v in tools_map.values()]
        all_specs = [v["schema"] for v in tools_map.values() if v.get("schema")]

        tools_map["__all__"] = {"call": all_calls, "schema": all_specs}
        return tools_map

    def get_all_available_tools(self) -> List[str]:
        """List all available LLM tools for the chatbot."""
        return [
            "llm_call",
            "extract_json_from_llm_response",
            "format_llm_response",
            "get_codebase_insights",
            "analyze_dependencies",
            "analyze_health_metrics",
            "get_security_analysis",
            "generate_final_report",
            "extract_intent_from_prompt",
        ]


class MockLLMClient:
    """Mock client for testing and offline development."""

    def generate_response(self, prompt: str) -> LLMResponse:
        content = (
            "This is a mock response for testing. "
            "In production, QGenie would provide detailed, professional analysis."
        )
        return LLMResponse(
            content=content,
            provider="mock",
            model="mock-model",
            tokens_used=len(content.split()),
            cost_estimate=0.0,
            response_time=0.05,
        )


def create_llm_integration(
    provider: str = "qgenie",
    model: str = "qgenie-default-model",
    api_key: Optional[str] = None,
    env_config: Optional[EnvConfig] = None,
    **kwargs,
) -> LLMIntegrations:
    """
    Factory function to create LLM integration (QGenie or Mock).
    """
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        provider_enum = LLMProvider.QGENIE if QGENIE_AVAILABLE else LLMProvider.MOCK

    config = LLMConfig(
        provider=provider_enum,
        model=model,
        api_key=api_key,
        **kwargs,
    )
    return LLMIntegrations(config, env_config)


def create_enhanced_llm_tools(env_config: Optional[EnvConfig] = None) -> LLMIntegrations:
    """
    Create enhanced LLM tools compatible with existing LLMTools interface.
    """
    return LLMIntegrations(env_config=env_config)