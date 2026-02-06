import os
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from statistics import median
import fnmatch
import math

from utils.common.llm_tools import LLMTools
from utils.parsers.env_parser import EnvConfig
from .analyzers.dependency_analyzer import DependencyAnalyzer, AnalyzerConfig
from .core.metrics_calculator import MetricsCalculator


class CodebaseAnalysisAgent:
    """
    LLM-Enhanced Agent for C/C++ codebase analysis: ingest, dependency graphing, metrics, and intelligent reporting.

    Focus:
    - C/C++ only (.c/.cc/.cpp/.cxx and headers .h/.hh/.hpp/.hxx)
    - Analyzes includes for dependency graph and external references
    - Uses LLM for intelligent analysis and recommendations
    - Computes dependency, quality (ScanBan-aligned), complexity, maintainability,
      documentation (Doxygen-centric), test coverage, security, and overall health
    """

    C_EXTS = {".c", ".cc", ".cpp", ".cxx"}
    H_EXTS = {".h", ".hh", ".hpp", ".hxx"}
    DEFAULT_EXTS = sorted(list(C_EXTS | H_EXTS))

    # Reasonable defaults for chunking large JSON contexts that go into prompts
    CHUNK_MAX_ITEMS_DEPENDENCY = 200
    CHUNK_MAX_ITEMS_DOCS = 200
    CHUNK_MAX_ITEMS_MOD_PLAN = 200
    CHUNK_MAX_ITEMS_VALIDATION = 200
    CHUNK_MAX_ITEMS_HEALTH = 200
    CHUNK_MAX_ITEMS_FINAL = 200

    def __init__(
        self,
        codebase_path: str,
        output_dir: str,
        file_extensions: Optional[List[str]] = None,
        max_files: int = 10000,
        exclude_dirs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        style_sample_size: int = 50,
        # New parameters:
        enable_llm: bool = True,
        env_config: Optional[Dict[str, Any]] = None,
        llm_tools: Optional["LLMTools"] = None,
    ):
        """
        Initialize the LLM-enhanced codebase analysis agent.
        """
        self.codebase_path = Path(codebase_path).resolve()
        self.file_extensions = file_extensions or self.DEFAULT_EXTS
        self.max_files = max_files
        self.style_sample_size = max(1, style_sample_size)
        self.output_dir = output_dir

        # Exclusions
        self.exclude_dirs = set(
            exclude_dirs
            or [
                ".git",
                "build",
                "dist",
                ".idea",
                ".vscode",
                ".tox",
                "node_modules",
                "third_party",
                "__pycache__",
                ".mypy_cache",
                ".pytest_cache",
            ]
        )
        self.exclude_globs = exclude_globs or []

        # Initialize LLM tools (allow injection, else default)
        self.llm_tools = llm_tools or LLMTools()

        # Initialize analyzers
        analyzer_config = AnalyzerConfig(
            project_root=str(self.codebase_path),
            ignore_dirs=[str(d) for d in self.exclude_dirs],
        )
        self.dependency_analyzer = DependencyAnalyzer(analyzer_config)

        # Derive optional project_root and debug from env_config if present
        project_root = None
        debug = False

        if isinstance(env_config, dict):
            project_root = env_config.get("project_root")
            debug = bool(env_config.get("debug", False))
        else:
            project_root = getattr(env_config, "project_root", None)
            debug = bool(getattr(env_config, "debug", False))

        if project_root is None:
            project_root = str(self.codebase_path)

        # Initialize metrics calculator with the same codebase context
        self.metrics_calculator = MetricsCalculator(
            codebase_path=str(self.codebase_path),
            output_dir=self.output_dir,
            project_root=project_root,
            debug=debug,
        )

        # Optional feature flag if you need to disable LLM externally
        self.enable_llm = enable_llm
        self.env_config = env_config or {}

        # Workflow artifacts
        self.summary: Dict = {}
        self.dependency_graph: Dict[str, Dict] = {}
        self.documentation: Dict[str, Dict] = {}
        self.modularization_plan: Dict[str, Dict] = {}
        self.validation_report: Dict[str, Dict] = {}
        self.health_metrics: Dict[str, Dict] = {}
        self.final_report: Dict[str, Dict] = {}
        self.errors: List[Dict] = []

        # Cache built during ingest
        self._file_cache: List[Dict] = []
        self._internal_c_cpp_modules: Set[str] = set()

    # --- Generic Chunking Helpers -------------------------------------------------

    def _chunk_list(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Yield successive chunks from a list without changing item content.
        """
        if not items or chunk_size <= 0:
            yield items
            return
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def _llm_batch_process(
        self,
        items: List[Any],
        chunk_size: int,
        build_prompt_from_chunk,
        parse_llm_response,
        merge_partial_results,
        context_stage: str,
    ) -> Any:
        """
        Generic helper to call LLM over multiple chunks of items while
        preserving prompt/logic semantics.

        - build_prompt_from_chunk(chunk) -> str (prompt string)
        - parse_llm_response(response_str) -> partial_result
        - merge_partial_results(accum, partial) -> accum
        """
        result = None
        for chunk in self._chunk_list(items, chunk_size):
            if not chunk:
                continue
            try:
                prompt = build_prompt_from_chunk(chunk)
                response = self.llm_tools.llm_call(prompt)
                partial = parse_llm_response(response)
                result = merge_partial_results(result, partial)
            except Exception as e:
                self.errors.append(
                    {
                        "stage": f"{context_stage}_chunk_llm_call",
                        "error": str(e),
                    }
                )
                # continue processing remaining chunks; result may be partial
        return result

    # --- Public Workflow Methods ---

    def ingest_codebase(self):
        """
        Ingest C/C++ codebase with LLM-enhanced analysis.
        """
        files = self._gather_files()
        cache = self._build_cache(files)

        # Basic analysis
        naming = self._analyze_naming(cache)
        includes = self._analyze_imports(cache)
        readme_notes = self._extract_readme_notes()
        style = self._analyze_style(cache)

        # LLM-enhanced analysis
        llm_insights = self._get_llm_codebase_insights(cache)

        file_metadata = [
            {
                "file_name": f["file_name"],
                "file_relative_path": f["file_relative_path"],
                "language": f["language"],
            }
            for f in cache
        ]

        self.summary = {
            "file_stats": {
                "total_files": len(cache),
                "file_types": dict(Counter([f["suffix"] for f in cache])),
                "files": file_metadata,
            },
            "naming_conventions": naming,
            "common_includes": includes,
            "readme_notes": readme_notes,
            "style": style,
            "llm_insights": llm_insights,
            "errors_count": len(self.errors),
        }
        return self.summary

    def build_dependency_graph(self):
        """
        Build C/C++ dependency graph with LLM analysis using DependencyAnalyzer.
        """
        cache = self._file_cache or self._build_cache(self._gather_files())

        # Use the specialized dependency analyzer
        self.dependency_graph = self.dependency_analyzer.build_graph(cache)

        # LLM-enhanced dependency analysis
        llm_dependency_analysis = self._get_llm_dependency_analysis(self.dependency_graph)
        self.dependency_graph["llm_analysis"] = llm_dependency_analysis

        return self.dependency_graph

    def generate_documentation(self):
        """Generate LLM-enhanced documentation analysis."""
        base_docs = self.dependency_analyzer.document_graph(self.dependency_graph)

        # LLM-enhanced documentation recommendations
        llm_doc_recommendations = self._get_llm_documentation_recommendations(base_docs)

        self.documentation = {
            "base_documentation": base_docs,
            "llm_recommendations": llm_doc_recommendations,
        }
        return self.documentation

    def plan_modularization(self):
        """Generate LLM-enhanced modularization plan."""
        base_plan = self.dependency_analyzer.propose_modularization(self.dependency_graph)

        # LLM-enhanced modularization recommendations
        llm_modularization = self._get_llm_modularization_plan(
            self.dependency_graph, base_plan
        )

        self.modularization_plan = {
            "base_plan": base_plan,
            "llm_enhanced_plan": llm_modularization,
        }
        return self.modularization_plan

    def validate_modularization(self):
        """Validate modularization with LLM insights."""
        base_validation = self.dependency_analyzer.validate_modularization(
            self.dependency_graph, self.modularization_plan.get("base_plan", {})
        )

        # LLM-enhanced validation
        llm_validation = self._get_llm_validation_insights(
            base_validation, self.modularization_plan
        )

        self.validation_report = {
            "base_validation": base_validation,
            "llm_validation": llm_validation,
        }
        return self.validation_report

    def calculate_health_metrics(self):
        """Calculate comprehensive health metrics using MetricsCalculator."""
        cache = self._file_cache or self._build_cache(self._gather_files())

        # Use the specialized metrics calculator
        metrics = self.metrics_calculator.calculate_all_metrics(
            cache, self.dependency_graph
        )

        # LLM-enhanced health analysis
        llm_health_analysis = self._get_llm_health_analysis(metrics)
        metrics["llm_analysis"] = llm_health_analysis

        self.health_metrics = metrics
        return metrics

    def generate_report(self):
        """Generate comprehensive LLM-enhanced report."""
        # Ensure all components are computed
        if not self.summary:
            self.ingest_codebase()
        if not self.dependency_graph:
            self.build_dependency_graph()
        if not self.documentation:
            self.generate_documentation()
        if not self.modularization_plan:
            self.plan_modularization()
        if not self.validation_report:
            self.validate_modularization()
        if not self.health_metrics:
            self.calculate_health_metrics()

        # Generate LLM-enhanced final report
        llm_final_report = self._generate_llm_final_report()

        self.final_report = {
            "summary": self.summary,
            "dependency_graph": self.dependency_graph,
            "documentation": self.documentation,
            "modularization_plan": self.modularization_plan,
            "validation_report": self.validation_report,
            "health_metrics": self.health_metrics,
            "llm_enhanced_report": llm_final_report,
            "errors": self.errors,
        }
        return self.final_report

    def analyze(self):
        """Run the full LLM-enhanced analysis workflow."""
        self.ingest_codebase()
        self.build_dependency_graph()
        self.generate_documentation()
        self.plan_modularization()
        self.validate_modularization()
        self.calculate_health_metrics()
        self.generate_report()

        try:
            self.save_modular_reports(self.output_dir)
        except Exception as e:
            self.errors.append(
                {
                    "stage": "auto_save_modular_reports",
                    "file": str(self.output_dir),
                    "error": str(e),
                }
            )

        return {
            "summary": self.summary,
            "dependency_graph": self.dependency_graph,
            "documentation": self.documentation,
            "modularization_plan": self.modularization_plan,
            "validation_report": self.validation_report,
            "health_metrics": self.health_metrics,
            "final_report": self.final_report,
        }

    # --- LLM Provider Helper ---

    def _get_llm_provider_name(self) -> str:
        """
        Safely retrieve the LLM provider name from LLMTools config.
        """
        cfg = getattr(self.llm_tools, "config", None)
        return getattr(cfg, "provider", "unknown")

    # --- LLM-Enhanced Analysis Methods ---

    def _get_llm_codebase_insights(self, cache: List[Dict]) -> Dict:
        """Get LLM insights about the codebase structure and patterns."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            codebase_summary = self._prepare_codebase_summary(cache)

            prompt = f"""
                You are a senior C/C++ software architect performing a high-level assessment of a codebase.

                Codebase Summary (JSON):
                {json.dumps(codebase_summary, indent=2)}

                You will receive a summarized view of the repository (file distribution, languages, sample files, etc.). This summary is not exhaustive, so:

                Base your conclusions only on the given data and obvious inferences.
                Clearly mark anything that is speculative or based on weak evidence.
                Using the provided summary, produce a concise but professional assessment that covers:

                Architecture patterns identified

                Any apparent layering, modular boundaries, or feature grouping.
                Signs of monoliths vs. modular components.
                Any inferred patterns (e.g., client/server, plugin system, driver/core separation).
                Reference concrete evidence from the summary (e.g., directory names, file groupings).
                Code organization assessment

                How coherent the directory layout and naming conventions appear.
                Consistency of language and file usage (headers vs. sources, platform‑specific code, etc.).
                Notable hot spots or unusual concentrations of files.
                Potential design and modularization issues

                Potential coupling/ownership problems, and locations that seem risky or over‑centralized.
                Areas likely to cause change ripple (e.g., shared “utils,” god modules, or mixed responsibilities).
                Explicitly call out any risks related to future scalability or maintainability.
                Recommendations for improvement

                Short‑term: quick, low‑risk improvements (e.g., cleanup, simple reorganizations).
                Medium‑term: focused refactors or re‑modularization steps.
                Long‑term: structural or architectural evolution. For each recommendation, include:
                a clear description,
                suggested scope (e.g., “target module X / directory Y”),
                expected impact,
                an approximate priority (e.g., P1/P2/P3).
                Technology stack and tooling assessment

                Languages, build systems, and any visible frameworks or libraries (even if inferred).
                Potential gaps (e.g., missing tests, missing linting, missing CI/CD signals).
                Any risks or constraints tied to the current stack.
                Output format: Respond as a single JSON object with the following top‑level keys:

                architecture_patterns
                organization_assessment
                design_issues
                recommendations
                tech_stack
                Each key should contain a structured value (arrays/objects), not free‑form text blobs. Where possible, include:

                'evidence' fields describing what in the summary led you to the conclusion.
                'severity' or 'priority' fields for issues and recommendations. Avoid including any text outside the JSON object. 
                """
            response = self.llm_tools.llm_call(prompt)
            parsed = self._parse_llm_json_response(response, "codebase_insights")
            parsed["provider"] = self._get_llm_provider_name()
            return parsed

        except Exception as e:
            self.errors.append({"stage": "llm_codebase_insights", "error": str(e)})
            return {"error": str(e)}

    def _get_llm_dependency_analysis(self, dependency_graph: Dict) -> Dict:
        """Get LLM analysis of dependency patterns and issues."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            dep_summary = self._prepare_dependency_summary(dependency_graph)

            # Chunk sample_dependencies if it is large to keep prompt manageable
            sample_deps_items = list(dep_summary.get("sample_dependencies", {}).items())
            if len(sample_deps_items) <= self.CHUNK_MAX_ITEMS_DEPENDENCY:
                # Single call path, identical to original behavior
                prompt = f"""
                You are a C/C++ software architect analyzing dependency relationships.

                Dependency Summary (JSON):
                {json.dumps(dep_summary, indent=2)}

                You will receive a summarized dependency graph (internal vs external nodes, fan-in/fan-out, high-fan nodes, sample edges, etc.). This graph may be incomplete, so:

                Base your conclusions only on the provided data and reasonable, conservative inferences.
                Mark anything that is speculative or low-confidence explicitly.
                Using this information, provide a structured, professional analysis that covers:

                Dependency hotspots and problematic patterns

                Identify modules that appear to be hotspots (e.g., very high fan-in or fan-out, central “hub” modules).
                Describe any visible patterns such as:
                God modules or central orchestrators.
                Shared utility modules overused across multiple areas.
                Potential layering or cross-layer dependencies.
                For each hotspot, include:
                A short description,
                Why it is problematic,
                Any relevant metrics or evidence (e.g., fan-in/fan-out values, references to node names).
                Coupling issues and recommendations

                Identify signs of:
                Excessive coupling (tight coupling, bidirectional dependencies),
                Unclear ownership boundaries,
                Over-reliance on common utility modules.
                For each coupling issue, provide:
                Description,
                Evidence (e.g., “module X has fan-in=Y, fan-out=Z”),
                Potential impact (e.g., change ripple, testing difficulty),
                Concrete recommendations to reduce or manage coupling (e.g., introduce interfaces, split modules, reorganize include structure).
                Architectural violations and their impact

                Based on the graph hints (e.g., layer indices, external vs internal, naming), identify possible:
                Cross-layer calls that break intended architecture,
                Feature leakage (e.g., low-level modules depending on high-level modules),
                Cycles between modules that should be in different layers.
                For each suspected violation, describe:
                The violation and the nodes involved,
                Likely architectural principle being violated (e.g., layering, dependency inversion),
                Impact on maintainability, testability, and scalability.
                Refactoring opportunities (concise, actionable steps)

                Propose concrete, feasible refactorings to improve the dependency structure.
                For each opportunity, provide:
                A short title,
                Description of the change,
                Scope (e.g., modules or directories likely affected),
                Step-by-step, actionable refactoring plan (small, implementable steps),
                Expected benefits (e.g., reduced coupling, fewer cycles, clearer boundaries),
                Rough priority (P1/P2/P3) and estimated effort (S/M/L).
                Risk assessment and mitigation recommendations

                Summarize key risks emerging from the current dependency structure, such as:
                High change ripple risk,
                Fragile build or integration points,
                Areas where defects are likely to cluster,
                Risks to incremental modularization.
                For each risk, provide:
                Description,
                Likelihood (low/medium/high),
                Impact (low/medium/high/critical),
                Concrete mitigation actions (both immediate and medium-term),
                Any monitoring or metrics you recommend (e.g., tracking fan-in/fan-out, cycle counts over time).
                Output format: Respond as a single JSON object with the following top-level keys:

                hotspots
                coupling_issues
                violations
                refactoring_opportunities
                risk_assessment
                Each key should contain structured data (arrays/objects) rather than free-form text blobs. Where possible, include:

                'evidence' fields (e.g., node names, fan-in/fan-out values, or brief references to the summary),
                'priority' and 'impact' or 'severity' fields for issues and recommendations,
                'steps' fields for refactoring actions.
                Do not include any text outside the JSON object.
                """
                response = self.llm_tools.llm_call(prompt)
                parsed = self._parse_llm_json_response(response, "dependency_analysis")
                parsed["provider"] = self._get_llm_provider_name()
                return parsed

            # Chunked path for large sample_dependencies
            def build_prompt_from_chunk(chunk_items):
                chunk_dep_summary = dict(dep_summary)
                chunk_dep_summary["sample_dependencies"] = dict(chunk_items)
                return f"""
                You are a C/C++ software architect analyzing dependency relationships.

                Dependency Summary (JSON):
                {json.dumps(chunk_dep_summary, indent=2)}

                You will receive a summarized dependency graph (internal vs external nodes, fan-in/fan-out, high-fan nodes, sample edges, etc.). This graph may be incomplete, so:

                Base your conclusions only on the provided data and reasonable, conservative inferences.
                Mark anything that is speculative or low-confidence explicitly.
                Using this information, provide a structured, professional analysis that covers:

                Dependency hotspots and problematic patterns

                Identify modules that appear to be hotspots (e.g., very high fan-in or fan-out, central “hub” modules).
                Describe any visible patterns such as:
                God modules or central orchestrators.
                Shared utility modules overused across multiple areas.
                Potential layering or cross-layer dependencies.
                For each hotspot, include:
                A short description,
                Why it is problematic,
                Any relevant metrics or evidence (e.g., fan-in/fan-out values, references to node names).
                Coupling issues and recommendations

                Identify signs of:
                Excessive coupling (tight coupling, bidirectional dependencies),
                Unclear ownership boundaries,
                Over-reliance on common utility modules.
                For each coupling issue, provide:
                Description,
                Evidence (e.g., “module X has fan-in=Y, fan-out=Z”),
                Potential impact (e.g., change ripple, testing difficulty),
                Concrete recommendations to reduce or manage coupling (e.g., introduce interfaces, split modules, reorganize include structure).
                Architectural violations and their impact

                Based on the graph hints (e.g., layer indices, external vs internal, naming), identify possible:
                Cross-layer calls that break intended architecture,
                Feature leakage (e.g., low-level modules depending on high-level modules),
                Cycles between modules that should be in different layers.
                For each suspected violation, describe:
                The violation and the nodes involved,
                Likely architectural principle being violated (e.g., layering, dependency inversion),
                Impact on maintainability, testability, and scalability.
                Refactoring opportunities (concise, actionable steps)

                Propose concrete, feasible refactorings to improve the dependency structure.
                For each opportunity, provide:
                A short title,
                Description of the change,
                Scope (e.g., modules or directories likely affected),
                Step-by-step, actionable refactoring plan (small, implementable steps),
                Expected benefits (e.g., reduced coupling, fewer cycles, clearer boundaries),
                Rough priority (P1/P2/P3) and estimated effort (S/M/L).
                Risk assessment and mitigation recommendations

                Summarize key risks emerging from the current dependency structure, such as:
                High change ripple risk,
                Fragile build or integration points,
                Areas where defects are likely to cluster,
                Risks to incremental modularization.
                For each risk, provide:
                Description,
                Likelihood (low/medium/high),
                Impact (low/medium/high/critical),
                Concrete mitigation actions (both immediate and medium-term),
                Any monitoring or metrics you recommend (e.g., tracking fan-in/fan-out, cycle counts over time).
                Output format: Respond as a single JSON object with the following top-level keys:

                hotspots
                coupling_issues
                violations
                refactoring_opportunities
                risk_assessment
                Each key should contain structured data (arrays/objects) rather than free-form text blobs. Where possible, include:

                'evidence' fields (e.g., node names, fan-in/fan-out values, or brief references to the summary),
                'priority' and 'impact' or 'severity' fields for issues and recommendations,
                'steps' fields for refactoring actions.
                Do not include any text outside the JSON object.
                """

            def parse_llm_response(resp):
                return self._parse_llm_json_response(resp, "dependency_analysis_chunk")

            def merge_partial_results(accum, partial):
                if accum is None:
                    return partial
                if not isinstance(accum, dict) or not isinstance(partial, dict):
                    return accum
                for key in ["hotspots", "coupling_issues", "violations", "refactoring_opportunities", "risk_assessment"]:
                    if key in partial:
                        if key not in accum or not isinstance(accum.get(key), list):
                            accum[key] = []
                        if isinstance(partial[key], list):
                            accum[key].extend(partial[key])
                return accum

            merged = self._llm_batch_process(
                items=sample_deps_items,
                chunk_size=self.CHUNK_MAX_ITEMS_DEPENDENCY,
                build_prompt_from_chunk=build_prompt_from_chunk,
                parse_llm_response=parse_llm_response,
                merge_partial_results=merge_partial_results,
                context_stage="llm_dependency_analysis",
            ) or {}

            merged["provider"] = self._get_llm_provider_name()
            return merged

        except Exception as e:
            self.errors.append({"stage": "llm_dependency_analysis", "error": str(e)})
            return {"error": str(e)}

    def _get_llm_documentation_recommendations(self, base_docs: Dict) -> Dict:
        """Get LLM recommendations for documentation improvements."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            # Flatten base_docs keys for chunking if needed
            # Keep full behavior if size is moderate
            serialized = json.dumps(base_docs)
            if len(serialized) <= 8000:
                prompt = f"""
                You are a senior technical documentation and architecture expert assessing the documentation posture of a C/C++ codebase.

                You will receive a structured analysis of the codebase and its dependencies. Using only this information and conservative inferences:

                Documentation Analysis (JSON):
                {json.dumps(base_docs, indent=2)}

                Base your conclusions on the provided signals (e.g., presence/absence of comments, READMEs, headers, public APIs).
                Clearly distinguish between strong evidence and low-confidence inference where applicable.
                Your task: Provide a thorough, structured assessment and recommendations across the following focus areas:

                Critical missing documentation
                Documentation quality improvements
                API documentation priorities
                Developer onboarding documentation
                Architecture documentation needs
                For each focus area, produce detailed sub-entries that include:

                issue_description:

                What is missing, weak, inconsistent, or confusing?
                Where possible, reference concrete evidence (e.g., “few comments in core modules”, “no top-level README”, “public headers lack Doxygen comments”).
                proposed_solution_steps:

                A clear, stepwise plan that is both:
                human-actionable (what a documentation owner or tech lead should do), and
                agent-actionable (what an automation or coding agent could help generate or validate).
                Include sequencing where helpful (e.g., “Step 1: inventory modules…, Step 2: define documentation templates…”).
                architectural_or_systemic_assessment:

                How documentation gaps reflect deeper architectural or process issues (e.g., unclear ownership, lack of API boundaries, missing design records).
                Any systemic patterns (e.g., code-first culture with no design docs, inconsistent use of comments across modules).
                risks_and_concerns_if_not_addressed:

                Specific risks tied to these documentation gaps (e.g., onboarding time, defect rate, incorrect usage of APIs, difficulty refactoring).
                Provide a sense of severity and time horizon (short-term vs long-term impact).
                concrete_recommendations:

                Practical, prioritized recommendations for each area.
                Include priority (P1/P2/P3) and approximate effort (S/M/L) where possible.
                Be explicit about suggested outputs (e.g., “Create module-level README for X”, “Add Doxygen comments for public headers in Y”).
                testing_and_validation_plan:

                An actionable plan to verify documentation quality and completeness over time.
                Examples: documentation checklists in CI, lint/coverage for comments, periodic doc reviews, automated checks for missing Doxygen for public APIs.
                Include both short-term checks and ongoing process/quality gates.
                Focus areas and output format:

                You MUST respond as a single JSON object with the following top-level keys:

                critical_missing
                quality_improvements
                api_priorities
                onboarding_needs
                architecture_docs
                Each of these keys should contain an array of structured sub-entries. Each sub-entry MUST follow this structure (all fields required):

                {{ "issue_description": "string", "proposed_solution_steps": [ "step 1 ...", "step 2 ..." ], "architectural_or_systemic_assessment": "string", "risks_and_concerns_if_not_addressed": [ {{ "risk": "string", "severity": "low | medium | high | critical", "time_horizon": "short_term | medium_term | long_term" }} ], "concrete_recommendations": [ {{ "description": "string", "priority": "P1 | P2 | P3", "estimated_effort": "S | M | L" }} ], "testing_and_validation_plan": [ {{ "description": "string", "type": "one_time | recurring | ci_gate", "ownership": "engineering | documentation | shared" }} ], "evidence": [ "brief references to observed patterns, file types, or metrics that support this analysis" ], "confidence": "high | medium | low" }}

                Additional instructions:

                Be concise but specific; avoid vague statements like “improve docs” without location and impact.
                Where evidence is weak, set confidence = "low" and clearly note that it is inferred.
                Return ONLY the JSON object and no additional text.
            """
                response = self.llm_tools.llm_call(prompt)
                parsed = self._parse_llm_json_response(
                    response, "documentation_recommendations"
                )
                parsed["provider"] = self._get_llm_provider_name()
                parsed["analysis_type"] = "enhanced_llm_documentation"
                return parsed

            # Chunked path: if base_docs is large, we process per top-level key
            items = list(base_docs.items())

            def build_prompt_from_chunk(chunk_items):
                chunk_docs = dict(chunk_items)
                return f"""
                You are a senior technical documentation and architecture expert assessing the documentation posture of a C/C++ codebase.

                You will receive a structured analysis of the codebase and its dependencies. Using only this information and conservative inferences:

                Documentation Analysis (JSON):
                {json.dumps(chunk_docs, indent=2)}

                Base your conclusions on the provided signals (e.g., presence/absence of comments, READMEs, headers, public APIs).
                Clearly distinguish between strong evidence and low-confidence inference where applicable.
                Your task: Provide a thorough, structured assessment and recommendations across the following focus areas:

                Critical missing documentation
                Documentation quality improvements
                API documentation priorities
                Developer onboarding documentation
                Architecture documentation needs
                For each focus area, produce detailed sub-entries that include:

                issue_description:

                What is missing, weak, inconsistent, or confusing?
                Where possible, reference concrete evidence (e.g., “few comments in core modules”, “no top-level README”, “public headers lack Doxygen comments”).
                proposed_solution_steps:

                A clear, stepwise plan that is both:
                human-actionable (what a documentation owner or tech lead should do), and
                agent-actionable (what an automation or coding agent could help generate or validate).
                Include sequencing where helpful (e.g., “Step 1: inventory modules…, Step 2: define documentation templates…”).
                architectural_or_systemic_assessment:

                How documentation gaps reflect deeper architectural or process issues (e.g., unclear ownership, lack of API boundaries, missing design records).
                Any systemic patterns (e.g., code-first culture with no design docs, inconsistent use of comments across modules).
                risks_and_concerns_if_not_addressed:

                Specific risks tied to these documentation gaps (e.g., onboarding time, defect rate, incorrect usage of APIs, difficulty refactoring).
                Provide a sense of severity and time horizon (short-term vs long-term impact).
                concrete_recommendations:

                Practical, prioritized recommendations for each area.
                Include priority (P1/P2/P3) and approximate effort (S/M/L) where possible.
                Be explicit about suggested outputs (e.g., “Create module-level README for X”, “Add Doxygen comments for public headers in Y”).
                testing_and_validation_plan:

                An actionable plan to verify documentation quality and completeness over time.
                Examples: documentation checklists in CI, lint/coverage for comments, periodic doc reviews, automated checks for missing Doxygen for public APIs.
                Include both short-term checks and ongoing process/quality gates.
                Focus areas and output format:

                You MUST respond as a single JSON object with the following top-level keys:

                critical_missing
                quality_improvements
                api_priorities
                onboarding_needs
                architecture_docs
                Each of these keys should contain an array of structured sub-entries. Each sub-entry MUST follow this structure (all fields required):

                {{ "issue_description": "string", "proposed_solution_steps": [ "step 1 ...", "step 2 ..." ], "architectural_or_systemic_assessment": "string", "risks_and_concerns_if_not_addressed": [ {{ "risk": "string", "severity": "low | medium | high | critical", "time_horizon": "short_term | medium_term | long_term" }} ], "concrete_recommendations": [ {{ "description": "string", "priority": "P1 | P2 | P3", "estimated_effort": "S | M | L" }} ], "testing_and_validation_plan": [ {{ "description": "string", "type": "one_time | recurring | ci_gate", "ownership": "engineering | documentation | shared" }} ], "evidence": [ "brief references to observed patterns, file types, or metrics that support this analysis" ], "confidence": "high | medium | low" }}

                Additional instructions:

                Be concise but specific; avoid vague statements like “improve docs” without location and impact.
                Where evidence is weak, set confidence = "low" and clearly note that it is inferred.
                Return ONLY the JSON object and no additional text.
                """

            def parse_llm_response(resp):
                return self._parse_llm_json_response(
                    resp, "documentation_recommendations_chunk"
                )

            def merge_partial_results(accum, partial):
                if accum is None:
                    return partial
                if not isinstance(accum, dict) or not isinstance(partial, dict):
                    return accum
                for key in [
                    "critical_missing",
                    "quality_improvements",
                    "api_priorities",
                    "onboarding_needs",
                    "architecture_docs",
                ]:
                    if key in partial:
                        if key not in accum or not isinstance(accum.get(key), list):
                            accum[key] = []
                        if isinstance(partial[key], list):
                            accum[key].extend(partial[key])
                return accum

            merged = self._llm_batch_process(
                items=items,
                chunk_size=self.CHUNK_MAX_ITEMS_DOCS,
                build_prompt_from_chunk=build_prompt_from_chunk,
                parse_llm_response=parse_llm_response,
                merge_partial_results=merge_partial_results,
                context_stage="llm_documentation_recommendations",
            ) or {}

            merged["provider"] = self._get_llm_provider_name()
            merged["analysis_type"] = "enhanced_llm_documentation"
            return merged

        except Exception as e:
            self.errors.append(
                {"stage": "llm_documentation_recommendations", "error": str(e)}
            )
            return {"error": str(e)}

    def _get_llm_modularization_plan(
        self, dependency_graph: Dict, base_plan: Dict
    ) -> Dict:
        """Get LLM-enhanced modularization recommendations."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            context = {
                "dependency_graph_summary": self._prepare_dependency_summary(
                    dependency_graph
                ),
                "base_plan": base_plan,
            }

            serialized = json.dumps(context)
            if len(serialized) <= 8000:
                prompt = f"""
                You are a senior C/C++ software architect refining and strengthening a modularization plan for a large codebase.

                You are given contextual information in JSON form. This may include:

                Current and proposed module/group boundaries.
                Dependency graphs and coupling metrics.
                Known cycles, cross-layer violations, and hotspots.
                Prior refactoring attempts, modularization proposals, and validation results.
                Context (JSON): {json.dumps(context, indent=2)}

                Instructions:

                Using ONLY the information in the context and conservative inferences, design an enhanced modularization strategy that is both:

                Architecturally sound and clearly communicated for humans, and
                Highly structured and actionable for automation and coding agents.
                For each major theme below, provide a structured analysis and plan that includes:

                Issue description (current structural/modular problems).
                Proposed solutions (stepwise, human- and agent-actionable).
                Architectural assessment and target module boundaries.
                Risks and concerns (with mitigations).
                Strategic recommendations (how to sequence and govern changes).
                Actionable testing/validation plan for modularization changes.
                The major themes are:

                Module boundaries (what the modules are and how they should be shaped).
                Interface design (how modules communicate and where to introduce/clean up interfaces).
                Migration strategy (how to move from current to target structure safely).
                Risk mitigation (how to reduce and manage risks during modularization).
                Testing and validation strategy (how to verify the modularization at each step).
                Output format:

                You MUST respond as a single JSON object with the following top-level keys:

                module_boundaries
                interface_design
                migration_strategy
                risk_mitigation
                testing_strategy
                Each of these keys must contain structured, machine-readable sub-elements, not free-form text blobs.

                Additional requirements:

                Always tie issues and recommendations to specific evidence or structures from the Context where possible (modules, groups, dependencies, metrics, validation results).
                Mark low-confidence inferences explicitly using a "confidence": "low" field where relevant.
                Avoid generic or boilerplate advice; keep recommendations and plans clearly grounded in the given context.
                Return ONLY the JSON object with the exact top-level keys:
                module_boundaries
                interface_design
                migration_strategy
                risk_mitigation
                testing_strategy and no extra text.
                """
                response = self.llm_tools.llm_call(prompt)
                parsed = self._parse_llm_json_response(response, "modularization_plan")
                parsed["provider"] = self._get_llm_provider_name()
                parsed["analysis_type"] = "enhanced_llm_modularization"
                return parsed

            # Chunked path: process context in pieces (per base_plan entry)
            base_plan_items = list(base_plan.items())

            def build_prompt_from_chunk(chunk_items):
                chunk_context = {
                    "dependency_graph_summary": context["dependency_graph_summary"],
                    "base_plan": dict(chunk_items),
                }
                return f"""
                You are a senior C/C++ software architect refining and strengthening a modularization plan for a large codebase.

                You are given contextual information in JSON form. This may include:

                Current and proposed module/group boundaries.
                Dependency graphs and coupling metrics.
                Known cycles, cross-layer violations, and hotspots.
                Prior refactoring attempts, modularization proposals, and validation results.
                Context (JSON): {json.dumps(chunk_context, indent=2)}

                Instructions:

                Using ONLY the information in the context and conservative inferences, design an enhanced modularization strategy that is both:

                Architecturally sound and clearly communicated for humans, and
                Highly structured and actionable for automation and coding agents.
                For each major theme below, provide a structured analysis and plan that includes:

                Issue description (current structural/modular problems).
                Proposed solutions (stepwise, human- and agent-actionable).
                Architectural assessment and target module boundaries.
                Risks and concerns (with mitigations).
                Strategic recommendations (how to sequence and govern changes).
                Actionable testing/validation plan for modularization changes.
                The major themes are:

                Module boundaries (what the modules are and how they should be shaped).
                Interface design (how modules communicate and where to introduce/clean up interfaces).
                Migration strategy (how to move from current to target structure safely).
                Risk mitigation (how to reduce and manage risks during modularization).
                Testing and validation strategy (how to verify the modularization at each step).
                Output format:

                You MUST respond as a single JSON object with the following top-level keys:

                module_boundaries
                interface_design
                migration_strategy
                risk_mitigation
                testing_strategy
                Each of these keys must contain structured, machine-readable sub-elements, not free-form text blobs.

                Additional requirements:

                Always tie issues and recommendations to specific evidence or structures from the Context where possible (modules, groups, dependencies, metrics, validation results).
                Mark low-confidence inferences explicitly using a "confidence": "low" field where relevant.
                Avoid generic or boilerplate advice; keep recommendations and plans clearly grounded in the given context.
                Return ONLY the JSON object with the exact top-level keys:
                module_boundaries
                interface_design
                migration_strategy
                risk_mitigation
                testing_strategy and no extra text.
                """

            def parse_llm_response(resp):
                return self._parse_llm_json_response(
                    resp, "modularization_plan_chunk"
                )

            def merge_partial_results(accum, partial):
                if accum is None:
                    return partial
                if not isinstance(accum, dict) or not isinstance(partial, dict):
                    return accum
                for key in [
                    "module_boundaries",
                    "interface_design",
                    "migration_strategy",
                    "risk_mitigation",
                    "testing_strategy",
                ]:
                    if key in partial:
                        if key not in accum or not isinstance(accum.get(key), list):
                            accum[key] = []
                        if isinstance(partial[key], list):
                            accum[key].extend(partial[key])
                return accum

            merged = self._llm_batch_process(
                items=base_plan_items,
                chunk_size=self.CHUNK_MAX_ITEMS_MOD_PLAN,
                build_prompt_from_chunk=build_prompt_from_chunk,
                parse_llm_response=parse_llm_response,
                merge_partial_results=merge_partial_results,
                context_stage="llm_modularization_plan",
            ) or {}

            merged["provider"] = self._get_llm_provider_name()
            merged["analysis_type"] = "enhanced_llm_modularization"
            return merged

        except Exception as e:
            self.errors.append({"stage": "llm_modularization_plan", "error": str(e)})
            return {"error": str(e)}

    def _get_llm_validation_insights(
        self, base_validation: Dict, modularization_plan: Dict
    ) -> Dict:
        """Get LLM insights on validation results."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            context = {
                "validation_results": base_validation,
                "modularization_plan": modularization_plan,
            }

            serialized = json.dumps(context)
            if len(serialized) <= 8000:
                prompt = f"""
"You are a senior C/C++ software quality and architecture expert validating a modularization plan.

You are given a validation context in JSON form, which may include:

Module/group definitions and their intended boundaries.
Dependency analysis results (including cycles, fan-in/fan-out, cross-layer links).
Build and test validation results (success/failures, errors, warnings).
Any prior refactoring or modularization recommendations that were applied or tested.
Validation Context (JSON): {json.dumps(context, indent=2)}

Instructions: Using ONLY the information in the validation context and conservative inferences:

Interpret validation results:
...
Return ONLY the JSON object with the exact top-level keys:
interpretation
critical_issues
dependency_fixes
strategy_improvements
success_criteria
risks
testing_plan and no extra text. "
"""
                response = self.llm_tools.llm_call(prompt)
                parsed = self._parse_llm_json_response(response, "validation_insights")
                parsed["provider"] = self._get_llm_provider_name()
                parsed["analysis_type"] = "enhanced_llm_validation"
                return parsed

            # Chunked path: process per validation result subset
            validation_items = list(base_validation.items())

            def build_prompt_from_chunk(chunk_items):
                chunk_validation = dict(chunk_items)
                chunk_context = {
                    "validation_results": chunk_validation,
                    "modularization_plan": modularization_plan,
                }
                return f"""
"You are a senior C/C++ software quality and architecture expert validating a modularization plan.

You are given a validation context in JSON form, which may include:

Module/group definitions and their intended boundaries.
Dependency analysis results (including cycles, fan-in/fan-out, cross-layer links).
Build and test validation results (success/failures, errors, warnings).
Any prior refactoring or modularization recommendations that were applied or tested.
Validation Context (JSON): {json.dumps(chunk_context, indent=2)}

Instructions: Using ONLY the information in the validation context and conservative inferences:

Interpret validation results:
...
Return ONLY the JSON object with the exact top-level keys:
interpretation
critical_issues
dependency_fixes
strategy_improvements
success_criteria
risks
testing_plan and no extra text. "
"""

            def parse_llm_response(resp):
                return self._parse_llm_json_response(
                    resp, "validation_insights_chunk"
                )

            def merge_partial_results(accum, partial):
                if accum is None:
                    return partial
                if not isinstance(accum, dict) or not isinstance(partial, dict):
                    return accum
                for key in [
                    "interpretation",
                    "critical_issues",
                    "dependency_fixes",
                    "strategy_improvements",
                    "success_criteria",
                    "risks",
                    "testing_plan",
                ]:
                    if key in partial:
                        if key not in accum or not isinstance(accum.get(key), list):
                            accum[key] = []
                        if isinstance(partial[key], list):
                            accum[key].extend(partial[key])
                return accum

            merged = self._llm_batch_process(
                items=validation_items,
                chunk_size=self.CHUNK_MAX_ITEMS_VALIDATION,
                build_prompt_from_chunk=build_prompt_from_chunk,
                parse_llm_response=parse_llm_response,
                merge_partial_results=merge_partial_results,
                context_stage="llm_validation_insights",
            ) or {}

            merged["provider"] = self._get_llm_provider_name()
            merged["analysis_type"] = "enhanced_llm_validation"
            return merged

        except Exception as e:
            self.errors.append({"stage": "llm_validation_insights", "error": str(e)})
            return {"error": str(e)}

    def _prepare_metrics_summary(self, metrics: Dict) -> Dict:
        """Prepare metrics summary for LLM analysis."""
        summary = {}
        for key, value in metrics.items():
            if key == "llm_analysis":
                continue
            if isinstance(value, dict):
                summary[key] = {
                    "score": value.get("score", 0),
                    "grade": value.get("grade", "F"),
                    "issues_count": len(value.get("issues", [])),
                    "top_issues": value.get("issues", [])[:3],
                }
        return summary

    def _prepare_metrics_summary_for_llm(self, metrics: Dict) -> Dict:
        """
        Prepare a condensed view of metrics for the health LLM call.
        This keeps the existing _prepare_metrics_summary behavior but adds
        an extra safeguard: strips very large arrays under each metric.
        """
        base = self._prepare_metrics_summary(metrics)
        # base is already condensed, but we keep function in case future metrics add arrays
        for k, v in base.items():
            if isinstance(v, dict):
                for sub_k, sub_v in list(v.items()):
                    if isinstance(sub_v, list) and len(sub_v) > 50:
                        v[sub_k] = sub_v[:50]
        return base

    def _get_llm_health_analysis(self, metrics: Dict) -> Dict:
        """Get comprehensive LLM analysis of health metrics."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            metrics_summary = self._prepare_metrics_summary_for_llm(metrics)
            serialized = json.dumps(metrics_summary)
            if len(serialized) <= 8000:
                prompt = f"""
                You are a senior C/C++ code quality and architecture expert analyzing a comprehensive health assessment for a large codebase.

                You are given aggregated health metrics in JSON form, which may include:
                ...
                Return ONLY the JSON object with the exact top-level keys:
                overall_assessment
                critical_areas
                improvement_roadmap
                risk_mitigation
                maintenance_recommendations
                productivity_impact and no extra text. 

"""
                response = self.llm_tools.llm_call(prompt)
                parsed = self._parse_llm_json_response(response, "health_analysis")
                parsed["provider"] = self._get_llm_provider_name()
                return parsed

            # Chunked path: process metrics summary per metric key
            items = list(metrics_summary.items())

            def build_prompt_from_chunk(chunk_items):
                chunk_summary = dict(chunk_items)
                return f"""
                You are a senior C/C++ code quality and architecture expert analyzing a comprehensive health assessment for a large codebase.

                You are given aggregated health metrics in JSON form, which may include:
                ...
                Metrics Summary (JSON):
                {json.dumps(chunk_summary, indent=2)}

                Return ONLY the JSON object with the exact top-level keys:
                overall_assessment
                critical_areas
                improvement_roadmap
                risk_mitigation
                maintenance_recommendations
                productivity_impact and no extra text. 

"""

            def parse_llm_response(resp):
                return self._parse_llm_json_response(resp, "health_analysis_chunk")

            def merge_partial_results(accum, partial):
                if accum is None:
                    return partial
                if not isinstance(accum, dict) or not isinstance(partial, dict):
                    return accum
                for key in [
                    "overall_assessment",
                    "critical_areas",
                    "improvement_roadmap",
                    "risk_mitigation",
                    "maintenance_recommendations",
                    "productivity_impact",
                ]:
                    if key in partial:
                        if key not in accum or not isinstance(accum.get(key), list):
                            accum[key] = []
                        if isinstance(partial[key], list):
                            accum[key].extend(partial[key])
                return accum

            merged = self._llm_batch_process(
                items=items,
                chunk_size=self.CHUNK_MAX_ITEMS_HEALTH,
                build_prompt_from_chunk=build_prompt_from_chunk,
                parse_llm_response=parse_llm_response,
                merge_partial_results=merge_partial_results,
                context_stage="llm_health_analysis",
            ) or {}

            merged["provider"] = self._get_llm_provider_name()
            return merged

        except Exception as e:
            self.errors.append({"stage": "llm_health_analysis", "error": str(e)})
            return {"error": str(e)}

    def _generate_llm_final_report(self) -> Dict:
        """Generate comprehensive LLM-enhanced final report."""
        if not self.enable_llm or not self.llm_tools:
            return {"status": "LLM analysis disabled"}

        try:
            report_context = {
                "summary": self.summary,
                "dependency_analysis": self.dependency_graph.get(
                    "llm_analysis", {}
                ),
                "documentation": self.documentation.get("llm_recommendations", {}),
                "modularization": self.modularization_plan.get(
                    "llm_enhanced_plan", {}
                ),
                "validation": self.validation_report.get("llm_validation", {}),
                "health_metrics": self.health_metrics.get("llm_analysis", {}),
            }

            serialized = json.dumps(report_context)
            if len(serialized) <= 8000:
                prompt = f"""
                You are a senior C/C++ software architect preparing an executive-level final report for a large codebase.
                ...
                Return ONLY the JSON object with the exact top-level keys:
                executive_summary
                immediate_actions
                strategic_recommendations
                risk_assessment
                success_metrics
                resource_requirements and no extra text.
"""
                response = self.llm_tools.llm_call(prompt)
                parsed = self._parse_llm_json_response(response, "final_report")
                parsed["provider"] = self._get_llm_provider_name()
                return parsed

            # Chunked path: process report_context by top-level sections
            items = list(report_context.items())

            def build_prompt_from_chunk(chunk_items):
                chunk_context = dict(chunk_items)
                return f"""
                You are a senior C/C++ software architect preparing an executive-level final report for a large codebase.

                You are given a context in JSON form which may include summary, dependency analysis, documentation recommendations, modularization plan, validation insights, and health metrics.

                Context (JSON):
                {json.dumps(chunk_context, indent=2)}
                ...
                Return ONLY the JSON object with the exact top-level keys:
                executive_summary
                immediate_actions
                strategic_recommendations
                risk_assessment
                success_metrics
                resource_requirements and no extra text.
"""

            def parse_llm_response(resp):
                return self._parse_llm_json_response(resp, "final_report_chunk")

            def merge_partial_results(accum, partial):
                if accum is None:
                    return partial
                if not isinstance(accum, dict) or not isinstance(partial, dict):
                    return accum
                for key in [
                    "executive_summary",
                    "immediate_actions",
                    "strategic_recommendations",
                    "risk_assessment",
                    "success_metrics",
                    "resource_requirements",
                ]:
                    if key in partial:
                        if key not in accum or not isinstance(accum.get(key), list):
                            accum[key] = []
                        if isinstance(partial[key], list):
                            accum[key].extend(partial[key])
                return accum

            merged = self._llm_batch_process(
                items=items,
                chunk_size=self.CHUNK_MAX_ITEMS_FINAL,
                build_prompt_from_chunk=build_prompt_from_chunk,
                parse_llm_response=parse_llm_response,
                merge_partial_results=merge_partial_results,
                context_stage="llm_final_report",
            ) or {}

            merged["provider"] = self._get_llm_provider_name()
            return merged

        except Exception as e:
            self.errors.append({"stage": "llm_final_report", "error": str(e)})
            return {"error": str(e)}

    # --- Helper Methods for LLM Integration ---

    def _prepare_codebase_summary(self, cache: List[Dict]) -> Dict:
        """Prepare a concise summary of the codebase for LLM analysis."""
        file_types = Counter([f["suffix"] for f in cache])
        languages = Counter([f["language"] for f in cache])

        sample_files = [
            {
                "path": f["file_relative_path"],
                "type": f["suffix"],
                "size": len(f.get("source", "")),
            }
            for f in cache[:20]
        ]

        return {
            "total_files": len(cache),
            "file_types": dict(file_types),
            "languages": dict(languages),
            "sample_files": sample_files,
            "codebase_path": str(self.codebase_path),
        }

    def _prepare_dependency_summary(self, dependency_graph: Dict) -> Dict:
        """Prepare dependency graph summary for LLM analysis."""
        if not dependency_graph:
            return {}

        clean_graph = {k: v for k, v in dependency_graph.items() if k != "llm_analysis"}

        internal_nodes = [
            k for k, v in clean_graph.items() if not v.get("external", False)
        ]
        external_nodes = [k for k, v in clean_graph.items() if v.get("external", False)]

        fan_out = {
            node: len(meta.get("dependencies", []))
            for node, meta in clean_graph.items()
        }
        fan_in = Counter()
        for node, meta in clean_graph.items():
            for dep in meta.get("dependencies", []):
                fan_in[dep] += 1

        return {
            "total_nodes": len(clean_graph),
            "internal_nodes": len(internal_nodes),
            "external_nodes": len(external_nodes),
            "high_fan_out": [(k, v) for k, v in fan_out.items() if v > 5][:10],
            "high_fan_in": fan_in.most_common(10),
            "sample_dependencies": dict(list(clean_graph.items())[:10]),
        }

    def _parse_llm_json_response(self, response: str, context: str) -> Dict:
        """Parse LLM JSON response with error handling."""
        try:
            clean_response = self.llm_tools.extract_json_from_llm_response(response)
            return json.loads(clean_response)
        except Exception as e:
            self.errors.append(
                {
                    "stage": f"parse_llm_response_{context}",
                    "error": f"Failed to parse LLM response: {str(e)}",
                    "response": response[:500],
                }
            )
            return {"error": f"Failed to parse LLM response for {context}"}

    # --- Utilities and Core C/C++ Ingest (kept from original) ---

    def _is_excluded(self, path: Path) -> bool:
        for part in path.parts:
            if part in self.exclude_dirs:
                return True
        rel_posix = str(path.relative_to(self.codebase_path).as_posix())
        for pattern in self.exclude_globs:
            if fnmatch.fnmatch(rel_posix, pattern):
                return True
        return False

    def _gather_files(self) -> List[Dict]:
        files = []
        if not self.codebase_path.exists():
            self.errors.append(
                {
                    "stage": "gather_files",
                    "file": str(self.codebase_path),
                    "error": "Path does not exist",
                }
            )
            return []

        print(f"DEBUG: Scanning directory: {self.codebase_path}")
        print(f"DEBUG: Looking for extensions: {self.file_extensions}")

        for root, dirs, filenames in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for fname in filenames:
                suffix = Path(fname).suffix.lower()
                if suffix not in [ext.lower() for ext in self.file_extensions]:
                    continue
                fpath = Path(root) / fname
                if self._is_excluded(fpath):
                    continue
                rel = fpath.relative_to(self.codebase_path)
                files.append(
                    {
                        "path_obj": fpath,
                        "file_name": fpath.name,
                        "file_relative_path": str(rel.as_posix()),
                        "suffix": suffix,
                    }
                )
                if len(files) >= self.max_files:
                    break
            if len(files) >= self.max_files:
                break

        print(f"DEBUG: Found {len(files)} files matching extensions")
        if files:
            print("DEBUG: Sample files found:")
            for f in files[:10]:
                print(f"  - {f['file_relative_path']} ({f['suffix']})")
        return files

    def _extract_readme_notes(self) -> str:
        candidates = [
            self.codebase_path / "README.md",
            self.codebase_path / "README",
            self.codebase_path / "README.txt",
        ]
        readme = next((p for p in candidates if p.exists()), None)
        if not readme:
            return ""
        try:
            content = readme.read_text(encoding="utf-8", errors="ignore")
            m = re.search(
                r"(#+\s*(Architecture|Usage)[\s\S]+?)(\n#+|\Z)",
                content,
                re.IGNORECASE,
            )
            if m:
                return m.group(1).strip()
            return content[:1000].strip()
        except Exception as e:
            self.errors.append(
                {"stage": "readme", "file": str(readme), "error": str(e)}
            )
            return ""

    def _analyze_style(self, cache: List[Dict]) -> Dict:
        return {"sampled_files": len(cache), "notes": ["Style analysis placeholder"]}

    @staticmethod
    def _c_cpp_module_key(rel_path: str) -> str:
        posix = Path(rel_path).as_posix()
        base, _sep, _ext = posix.rpartition(".")
        base = base if base else posix
        return base.replace("/", ".")

    def _build_cache(self, files: List[Dict]) -> List[Dict]:
        cache = []
        c_modules: Set[str] = set()

        for f in files:
            path_obj = f["path_obj"]
            rel_path = f["file_relative_path"]
            suffix = f["suffix"]
            try:
                with open(path_obj, "r", encoding="utf-8", errors="ignore") as fh:
                    source = fh.read()
            except Exception as e:
                self.errors.append(
                    {
                        "stage": "read_file",
                        "file": str(path_obj),
                        "error": str(e),
                    }
                )
                continue

            entry = {
                "path_obj": path_obj,
                "file_name": f["file_name"],
                "file_relative_path": rel_path,
                "suffix": suffix,
                "language": self._infer_language(suffix),
                "source": source,
                "module_key": None,
                "path": str(path_obj),
                "rel_path": str(rel_path),
                "size_bytes": len(source.encode("utf-8")),
            }

            try:
                if suffix in (self.C_EXTS | self.H_EXTS):
                    module_key = self._c_cpp_module_key(rel_path)
                    entry["module_key"] = module_key
                    if module_key:
                        c_modules.add(module_key)

                    # Extract includes for dependency analysis
                    includes = []
                    for line_num, line in enumerate(source.splitlines(), 1):
                        line = line.strip()
                        include_match = re.match(
                            r'^\s*#\s*include\s*[<"]([^">]+)[">]', line
                        )
                        if include_match:
                            included_file = include_match.group(1)
                            includes.append(
                                {
                                    "file": included_file,
                                    "line": line_num,
                                    "type": "system"
                                    if line.find("<") != -1
                                    else "local",
                                    "raw_line": line,
                                }
                            )
                    entry["includes"] = includes

            except Exception as e:
                self.errors.append(
                    {
                        "stage": "cache_build",
                        "file": str(path_obj),
                        "error": str(e),
                    }
                )
                continue

            cache.append(entry)

        self._file_cache = cache
        self._internal_c_cpp_modules = {m for m in c_modules if m}

        c_cpp_files = [
            f for f in cache if f["suffix"] in (self.C_EXTS | self.H_EXTS)
        ]
        source_files = [f for f in c_cpp_files if f["suffix"] in self.C_EXTS]
        header_files = [f for f in c_cpp_files if f["suffix"] in self.H_EXTS]

        print(f"DEBUG: Total files cached: {len(cache)}")
        print(f"DEBUG: C/C++ files found: {len(c_cpp_files)}")
        print(f"DEBUG: Source files (.c/.cpp): {len(source_files)}")
        print(f"DEBUG: Header files (.h/.hpp): {len(header_files)}")
        print(f"DEBUG: File extensions found: {set(f['suffix'] for f in cache)}")

        if c_cpp_files:
            print("DEBUG: Sample C/C++ files:")
            for f in c_cpp_files[:5]:
                print(
                    f"  - {f['file_relative_path']} ({f['suffix']}, {len(f.get('source', ''))} chars)"
                )

        return cache

    @staticmethod
    def _infer_language(suffix: str) -> str:
        suffix = suffix.lower()
        if suffix in {".c"}:
            return "c"
        elif suffix in {".cc", ".cpp", ".cxx"}:
            return "cpp"
        elif suffix in {".h", ".hh", ".hpp", ".hxx"}:
            return "header"
        return "unknown"

    def _analyze_naming(self, cache: List[Dict]) -> Dict:
        return {"function_case": "mixed", "examples": []}

    def _analyze_imports(self, cache: List[Dict]) -> List[Tuple[str, int]]:
        include_counter = Counter()
        for entry in cache:
            suffix = entry["suffix"]
            source = entry["source"]
            if suffix not in (self.C_EXTS | self.H_EXTS):
                continue
            try:
                for line in source.splitlines():
                    m = re.match(
                        r'#\s*include\s*[<"]([^">]+)[">]', line.strip()
                    )
                    if m:
                        include_counter[m.group(1)] += 1
            except Exception as e:
                self.errors.append(
                    {
                        "stage": "analyze_imports",
                        "file": entry["file_relative_path"],
                        "error": str(e),
                    }
                )
                continue
        return include_counter.most_common(10)

    # --- Visualization Methods ---

    @staticmethod
    def _sanitize_mermaid_id(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", name)

    @staticmethod
    def _unique_sanitized_ids(names: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        used: Dict[str, int] = {}
        for name in names:
            base = CodebaseAnalysisAgent._sanitize_mermaid_id(name)
            if base not in used:
                used[base] = 1
                mapping[name] = base
            else:
                used[base] += 1
                mapping[name] = f"{base}__{used[base]}"
        return mapping

    def generate_dependency_graph_visualization(
        self, output_path: str, max_modules: int = 50000
    ):
        dependency_graph = self.final_report.get("dependency_graph", {})
        modularization_plan = self.final_report.get("modularization_plan", {}).get(
            "base_plan", {}
        )

        nodes = [k for k in dependency_graph.keys() if k != "llm_analysis"][
            :max_modules
        ]
        sanitized_nodes = self._unique_sanitized_ids(nodes)

        split_modules = {
            m for m, v in modularization_plan.items() if v.get("action") == "split"
        }
        external_nodes = {
            n
            for n in nodes
            if dependency_graph.get(n, {}).get("external", False)
        }

        lines = [
            "graph TD",
            "  %% High-coupling modules (suggested for splitting) are colored red; external nodes are gray",
        ]

        for node in nodes:
            sid = sanitized_nodes[node]
            label = self._sanitize_mermaid_id(node)
            if node in split_modules:
                lines.append(f"  {sid}[{label}]:::split")
            elif node in external_nodes:
                lines.append(f"  {sid}[{label}]:::external")
            else:
                lines.append(f"  {sid}[{label}]")

        added_external_ids = set()
        for node in nodes:
            sid = sanitized_nodes[node]
            deps = dependency_graph.get(node, {}).get("dependencies", [])
            for dep in deps:
                if dep in sanitized_nodes:
                    lines.append(f"  {sid} --> {sanitized_nodes[dep]}")
                else:
                    dep_sid = self._sanitize_mermaid_id(dep)
                    if (
                        dep_sid not in sanitized_nodes.values()
                        and dep_sid not in added_external_ids
                    ):
                        lines.append(
                            f"  {dep_sid}[{dep_sid}]:::external"
                        )
                        added_external_ids.add(dep_sid)
                    lines.append(f"  {sid} --> {dep_sid}")

        lines.append(
            "  classDef split fill:#ffe6e6,stroke:#d33,stroke-width:2px;"
        )
        lines.append(
            "  classDef external fill:#eee,stroke:#999,stroke-width:1px;"
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            self.errors.append(
                {
                    "stage": "write_visualization",
                    "file": output_path,
                    "error": str(e),
                }
            )
            raise
        return output_path

    def generate_dependency_graph_visualization_split(
        self, output_dir: str, max_edges: int = 500
    ):
        dependency_graph = self.final_report.get("dependency_graph", {})
        modularization_plan = self.final_report.get("modularization_plan", {}).get(
            "base_plan", {}
        )
        split_modules = {
            m for m, v in modularization_plan.items() if v.get("action") == "split"
        }

        edges = []
        nodes = set()
        for node, meta in dependency_graph.items():
            if node == "llm_analysis":
                continue
            nodes.add(node)
            for dep in meta.get("dependencies", []):
                edges.append((node, dep))
                nodes.add(dep)

        output_paths = []

        for chunk_idx in range(0, len(edges), max_edges):
            chunk_edges = edges[chunk_idx : chunk_idx + max_edges]
            chunk_nodes = set()
            for src, dst in chunk_edges:
                chunk_nodes.add(src)
                chunk_nodes.add(dst)

            sanitized = self._unique_sanitized_ids(list(chunk_nodes))

            lines = [
                "graph TD",
                "  %% High-coupling modules (suggested for splitting) are colored red; external nodes are gray",
            ]

            for node in chunk_nodes:
                sid = sanitized[node]
                label = self._sanitize_mermaid_id(node)
                if node in split_modules:
                    lines.append(f"  {sid}[{label}]:::split")
                else:
                    if dependency_graph.get(node, {}).get("external", False):
                        lines.append(f"  {sid}[{label}]:::external")
                    else:
                        lines.append(f"  {sid}[{label}]")

            for src, dst in chunk_edges:
                src_id = sanitized.get(src, self._sanitize_mermaid_id(src))
                dst_id = sanitized.get(dst, self._sanitize_mermaid_id(dst))
                lines.append(f"  {src_id} --> {dst_id}")

            lines.append(
                "  classDef split fill:#ffe6e6,stroke:#d33,stroke-width:2px;"
            )
            lines.append(
                "  classDef external fill:#eee,stroke:#999,stroke-width:1px;"
            )

            try:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(
                    output_dir,
                    f"dependency_graph_part_{chunk_idx // max_edges + 1}.mmd",
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                output_paths.append(out_path)
            except Exception as e:
                self.errors.append(
                    {
                        "stage": "write_visualization_split",
                        "file": output_dir,
                        "error": str(e),
                    }
                )
                raise

        return output_paths

    # --- Report Generation and Persistence (Monolithic + Modular) ---

    @staticmethod
    def _convert_sets_to_lists(obj):
        if isinstance(obj, dict):
            return {k: CodebaseAnalysisAgent._convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, list):
            return [CodebaseAnalysisAgent._convert_sets_to_lists(i) for i in obj]
        else:
            return obj

    @staticmethod
    def _write_json_file(data: Any, path: str) -> str:
        """
        Helper to write JSON data to the specified path, converting sets to lists.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cleaned = CodebaseAnalysisAgent._convert_sets_to_lists(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        return path

    def save_report(self, path: str, filename: str = "health_report_llm.json") -> str:
        """
        Save the monolithic final report (for backward compatibility).
        """
        try:
            os.makedirs(path, exist_ok=True)
            report_path = os.path.join(path, filename)
            cleaned_report = CodebaseAnalysisAgent._convert_sets_to_lists(
                self.final_report
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_report, f, indent=2)
            return report_path
        except Exception as e:
            self.errors.append({"stage": "save_report", "file": path, "error": str(e)})
            raise

    def save_health_report(self, output_dir: Optional[str] = None) -> str:
        """
        Save the monolithic LLM health report as health_report_llm.json.
        """
        target_dir = output_dir or str(self.codebase_path)
        return self.save_report(target_dir, filename="health_report_llm.json")

    def save_modular_reports(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save modular JSON reports for each major analysis artifact and per-metric outputs.

        Outputs (all in output_dir or codebase root by default):
        - dependency_graph.json
        - documentation_report.json
        - modularization_plan.json
        - validation_report.json
        - health_metrics.json
        - dependency_report.json (if present in health_metrics)
        - quality_report.json
        - complexity_report.json
        - maintainability_report.json
        - documentation_metrics_report.json
        - test_coverage_report.json
        - security_report.json
        - overall_health_report.json
        """
        target_dir = output_dir or str(self.output_dir)
        os.makedirs(target_dir, exist_ok=True)
        written: Dict[str, str] = {}

        try:
            # Dependency graph (including LLM analysis)
            if self.dependency_graph:
                path = os.path.join(target_dir, "dependency_graph.json")
                written["dependency_graph"] = self._write_json_file(
                    self.dependency_graph, path
                )

            # Documentation (raw + LLM)
            if self.documentation:
                path = os.path.join(target_dir, "documentation_report.json")
                written["documentation"] = self._write_json_file(
                    self.documentation, path
                )

            # Modularization plan
            if self.modularization_plan:
                path = os.path.join(target_dir, "modularization_plan.json")
                written["modularization_plan"] = self._write_json_file(
                    self.modularization_plan, path
                )

            # Validation report
            if self.validation_report:
                path = os.path.join(target_dir, "validation_report.json")
                written["validation_report"] = self._write_json_file(
                    self.validation_report, path
                )

            # Full health metrics
            if self.health_metrics:
                path = os.path.join(target_dir, "health_metrics.json")
                written["health_metrics"] = self._write_json_file(
                    self.health_metrics, path
                )

                # Per-metric modular reports (if present)
                metric_map = {
                    "dependency": "dependency_report.json",
                    "quality": "quality_report.json",
                    "complexity": "complexity_report.json",
                    "maintainability": "maintainability_report.json",
                    "documentation": "documentation_metrics_report.json",
                    "test_coverage": "test_coverage_report.json",
                    "security": "security_report.json",
                    "overall_health": "overall_health_report.json",
                }
                for metric_key, filename in metric_map.items():
                    if metric_key in self.health_metrics:
                        metric_data = self.health_metrics.get(metric_key, {})
                        metric_path = os.path.join(target_dir, filename)
                        written[metric_key] = self._write_json_file(
                            metric_data, metric_path
                        )

        except Exception as e:
            self.errors.append(
                {"stage": "save_modular_reports", "file": target_dir, "error": str(e)}
            )
            raise

        return written

    def write_health_report(self, output_dir: Optional[str] = None) -> str:
        """
        For backward compatibility: run analysis (if needed) and write the monolithic report.
        """
        if not self.final_report or not self.final_report.get("health_metrics"):
            self.analyze()
        else:
            self.generate_report()
            # Also write modular reports when explicitly writing
            try:
                self.save_modular_reports(output_dir)
            except Exception as e:
                self.errors.append(
                    {
                        "stage": "write_modular_reports",
                        "file": str(output_dir or self.codebase_path),
                        "error": str(e),
                    }
                )
        return self.save_health_report(output_dir)

    def print_report(self):
        import pprint

        pprint.pprint(self.final_report)