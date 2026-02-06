# llm_tools.py

from __future__ import annotations
import json
import uuid
import re
from pathlib import Path
import numpy as np
from rich.pretty import pprint
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import HumanMessage
from qgenie.integrations.langchain import QGenieChat

from utils.parsers.env_parser import EnvConfig
from db.vectordb_wrapper import VectorDB

import logging
import sys
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # <-- add this to send logs to stdout
    force=True          # <-- ensure configuration is enforced, even if already set
)
logger = logging.getLogger(__name__)

class LLMTools:
    def __init__(self):
        self.env_config = EnvConfig()
        self.prompt_file_path = self.resolve_relative_path(self.env_config.get("CHAT_PROMPT_FILE_PATH"))
        self.vectordb = self.get_dvt_vectordb()
        self.logger = logger


    @staticmethod
    def get_repo_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @classmethod
    def resolve_relative_path(cls, relative_path: str | Path) -> Path:
        if relative_path is None:
            raise ValueError("CHAT_PROMPT_FILE_PATH is not set in environment/config.")
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return cls.get_repo_root() / p

    @staticmethod
    def is_uuid(value: str) -> bool:
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def is_full_report_request(user_input):
        full_report_keywords = [
            "all modules", "all", "trend", "all records", "summary"
        ]
        return any(keyword in user_input.lower() for keyword in full_report_keywords)

    
    def metadata_filtering(self, all_records, **criteria):
        """
        Returns records matching all given metadata/data criteria.
        Usage example: meta_filtering(all_records, device_under_test="DUT-01", channel=7)
        - all_records: list of dicts
        - criteria: key-value pairs to match (searches both 'metadata' and 'data' fields)
        """
        def match(rec):
            for key, val in criteria.items():
                meta_val = rec.get('metadata', {}).get(key)
                data_val = rec.get('data', {}).get(key)
                if meta_val == val or data_val == val:
                    continue
                return False
            return True
        return [rec for rec in all_records if match(rec)]
    
    def get_dvt_vectordb(self):
        return VectorDB(self.env_config)

    #####################################################
    #########retrieve_relevant_docs######################
    #####################################################
    def retrieve_relevant_docs(self, input_str, top_k=250):
        """
        Robust hybrid semantic/metadata retrieval.
        Supports comparison, multi-entity, and special formatting queries.
        Now with detailed debug logging.

        Note:
        - This function is intentionally agnostic to output_format (e.g. "table",
        "tabular_list", "summary"). It only retrieves the relevant documents
        based on intent/criteria/entities.
        - Use output_format from the returned intent object in downstream
        formatting/rendering logic.
        """
        logger.debug(f"[retrieve_relevant_docs] Called with input_str: {input_str}, top_k: {top_k}")

        if self.vectordb is None:
            logger.warning("[retrieve_relevant_docs] No vector store available!")
            return []

        # --- STEP 1: Extract structured intent ---
        try:
            intent_obj = self.extract_intent_from_prompt(input_str)
            logger.debug(f"[retrieve_relevant_docs] Extracted intent object: {intent_obj}")
        except Exception as e:
            logger.error(f"[retrieve_relevant_docs] Failed to parse intent from prompt. Error: {e}")
            intent_obj = {"intent": "retrieve", "criteria": {}}

        # Optional: log output_format for debugging (but don't act on it here)
        output_format = intent_obj.get("output_format", "summary")
        logger.debug(f"[retrieve_relevant_docs] output_format from intent: {output_format}")

        results = []

        # --- STEP 2: Handle multi-entity 'compare' queries ---
        if intent_obj.get("intent") == "compare" and "entities" in intent_obj:
            results = []
            logger.debug(f"[retrieve_relevant_docs] Handling 'compare' intent for entities: {intent_obj['entities']}")

            for entity in intent_obj["entities"]:
                # Remove fields_to_extract if present, keep only filter keys
                criteria = {k: v for k, v in entity.items() if k != "fields_to_extract"}
                query_str = " ".join([f"{k} {v}" for k, v in criteria.items()])
                logger.debug(
                    "[retrieve_relevant_docs] Compare entity criteria: %s, query_str: '%s'",
                    criteria, query_str
                )
                matched_docs = self._semantic_and_metadata_search(query_str, criteria, top_k=top_k)
                logger.debug(
                    "[retrieve_relevant_docs] Found %d docs for criteria %s",
                    len(matched_docs), criteria
                )
                # Append full docs list for each entity
                results.append(matched_docs)

            logger.info(f"[retrieve_relevant_docs] Compare query processed. Result length: {len(results)}")
            # Optionally return both docs and intent if your caller expects it:
            # return {"docs": results, "intent": intent_obj}
            return results

        # --- STEP 3: Handle single-entity 'retrieve' or default queries ---
        else:
            # Try to use structured criteria; fall back to heuristic criteria extraction if available
            criteria = intent_obj.get("criteria") or getattr(
                self, 'extract_criteria_from_prompt', lambda x: {}
            )(input_str)

            logger.debug(f"[retrieve_relevant_docs] Handling 'retrieve' intent with criteria: {criteria}")
            matched_docs = self._semantic_and_metadata_search(input_str, criteria, top_k=top_k)
            logger.debug(f"[retrieve_relevant_docs] Found {len(matched_docs)} docs matching criteria.")

            # If you later decide to prune by fields_to_extract, you can re-enable this:
            # if "fields_to_extract" in intent_obj:
            #     logger.debug(
            #         "[retrieve_relevant_docs] Pruning docs to fields: %s",
            #         intent_obj["fields_to_extract"]
            #     )
            #     matched_docs = self._prune_doc_fields(matched_docs, intent_obj["fields_to_extract"])

            results = matched_docs
            logger.info(f"[retrieve_relevant_docs] Retrieve query processed. Result length: {len(results)}")

        # --- STEP 4: Optional: Log if no docs were found ---
        if not results or (isinstance(results, list) and len(results) == 0):
            logger.warning(f"[retrieve_relevant_docs] No records found for user input: {input_str}")

        # Optionally, return intent_obj along with results so caller can use output_format later:
        # return {"docs": results, "intent": intent_obj}

        return results

    def _semantic_and_metadata_search(self, query_str, criteria, top_k=250):
        logger.debug(f"[_semantic_and_metadata_search] query_str: '{query_str}', criteria: {criteria}, top_k: {top_k}")
        docs = self.vectordb.retrieve(query_str, k=top_k)
        logger.debug(f"[_semantic_and_metadata_search] Retrieved {len(docs)} docs from vectordb.")
        results = []
        for doc in docs:
            meta = getattr(doc, "metadata", None)
            if not meta:
                continue
            matches = True
            for k, v in criteria.items():
                meta_val = meta.get(k) if isinstance(meta, dict) else getattr(meta, k, None)
                data_val = getattr(doc, 'data', {}).get(k) if hasattr(doc, 'data') else None
                if meta_val == v or data_val == v:
                    continue
                matches = False
                break
            if matches:
                results.append(doc)
        logger.debug(f"[_semantic_and_metadata_search] Filtered to {len(results)} matching docs.")
        return results
    
    def _prune_doc_fields(self, docs, fields_to_extract):
        """
        Returns documents with only the desired measurement fields (plus always metadata).
        """
        pruned = []
        for doc in docs:
            base = {"metadata": getattr(doc, "metadata", {})}
            data = getattr(doc, "data", {})
            base["data"] = {k: v for k, v in data.items() if k in fields_to_extract}
            pruned.append(base)
        return pruned


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
            '- \"retrieve\"   – Get information about specific entities (modules, files, components, branches, services, tests, etc.).\n'
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
        
    def update_markdown_prompt(self, md_filepath, input_str, records):
        try:
            with open(md_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            prompt_template = content.strip().replace('\r\n', '\n')
            prompt = prompt_template.format(
                records=records,
                input_str=input_str,
            )
            return prompt
        except Exception as e:
            logger.error(f"Error reading {md_filepath}: {e}")
            return None

    def llm_call(self, prompt, model=None):
        # LLM call separated
        try:
            model = QGenieChat(
                model=self.env_config.get("LLM_MODEL"),
                api_key=self.env_config.get("QGENIE_API_KEY"),
                timeout=15000
            )
            messages = [HumanMessage(content=prompt)]
            result = model.invoke(
                messages,
                max_tokens=15000,
                repetition_penalty=1.1,
                temperature=0.1,
                top_k=50,
                top_p=0.95,
            )
            return result.content
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return f"LLM invocation failed: {e}"
    import re

    def extract_json_from_llm_response(self, response: str) -> str:
        """
        Extracts the JSON object from an LLM response that may be wrapped in Markdown code block.
        """
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response)
        if match:
            return match.group(1).strip()
        return response.strip()  # fallback: assume raw JSON
    
    def format_llm_response(self, agent_response):
        try:
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
                    elif hasattr(last, "content"):
                        return last.content
                    else:
                        return str(last)
                last = agent_response[-1]
                if isinstance(last, dict):
                    return last.get("content", str(last))
                elif hasattr(last, "content"):
                    return last.content
                else:
                    return str(last)
            elif hasattr(agent_response, "role") and hasattr(agent_response, "content"):
                return agent_response.content
            elif isinstance(agent_response, str):
                return agent_response
            elif agent_response is None:
                return "No response."
            else:
                return f"Unknown response type: {type(agent_response)}"
        except Exception as e:
            return f"Error extracting answer: {str(e)}"

    def get_tools_map(
        self,
        names: list[
            str
        ],
    ):
        tool_mapping = {
            "retrieve_relevant_docs": self.retrieve_relevant_docs,
            "update_markdown_prompt": self.update_markdown_prompt,
            "llm_call": self.llm_call,
            "format_llm_response": self.format_llm_response
        }

        tools_map = {}
        for name in names:
            tool = tool_mapping[name]
            tools_map[name] = {}
            tools_map[name]["call"] = tool
            tools_map[name]["schema"] = convert_to_openai_tool(tool)

        all_tools = []
        all_specs = []
        for v in tools_map.values():
            all_tools.append(v["call"])
            all_specs.append(v["schema"])

        tools_map["__all__"] = {}
        tools_map["__all__"]["call"] = all_tools
        tools_map["__all__"]["schema"] = all_specs

        return tools_map

    def get_all_available_tools(self):
        return ["retrieve_relevant_docs", "update_markdown_prompt", "llm_call", "format_llm_response"]
