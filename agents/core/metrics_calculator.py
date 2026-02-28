"""
Health metrics calculation coordinator for Verilog/SystemVerilog HDL codebases (enhanced).

Orchestrates:
- 9 HDL-specific analyzers (quality, complexity, synthesis_safety, documentation, etc.)
- Deep static analysis adapters (Verilator, Verible, hierarchy analysis)
"""

from typing import Dict, List, Any, Optional
import json
import logging
import os

logger = logging.getLogger(__name__)

# HDL Analyzers
from agents.analyzers.quality_analyzer import QualityAnalyzer
from agents.analyzers.complexity_analyzer import ComplexityAnalyzer
from agents.analyzers.security_analyzer import SynthesisSafetyAnalyzer  # Renamed
from agents.analyzers.documentation_analyzer import DocumentationAnalyzer
from agents.analyzers.maintainability_analyzer import MaintainabilityAnalyzer
from agents.analyzers.test_coverage_analyzer import VerificationCoverageAnalyzer  # Renamed
from agents.analyzers.potential_deadlock_analyzer import CDCAnalyzer  # Renamed (Clock Domain Crossing)
from agents.analyzers.null_pointer_analyzer import UninitializedSignalAnalyzer  # Renamed
from agents.analyzers.memory_corruption_analyzer import SignalIntegrityAnalyzer  # Renamed

# HDL-specific adapters (optional — graceful degradation)
try:
    from agents.adapters import (
        UnusedModuleAdapter,  # Renamed from DeadCodeAdapter
        HDLComplexityAdapter,  # Renamed from ASTComplexityAdapter
        LintAdapter,  # Renamed from SecurityAdapter
        HierarchyAnalyzerAdapter,  # Renamed from CallGraphAdapter
        ModuleMetricsAdapter,  # Renamed from FunctionMetricsAdapter
        ExcelReportAdapter,
        DependencyGraphAdapter,  # NEW: HDL dependency analysis
    )
    ADAPTERS_AVAILABLE = True
except ImportError as _e:
    ADAPTERS_AVAILABLE = False
    DependencyGraphAdapter = None
    logger.info(f"HDL analysis adapters not available: {_e}")


class MetricsCalculator:
    """
    Coordinates calculation of all health metrics for Verilog/SystemVerilog HDL codebases.
    """

    def __init__(
        self,
        codebase_path: str,
        output_dir: str,
        project_root: Optional[str] = None,
        debug: bool = False,
        enable_adapters: bool = False,
    ):
        """Initialize metrics calculator with all HDL analyzers."""
        self.codebase_path = codebase_path
        self.project_root = project_root
        self.debug = debug
        self.output_dir = output_dir

        # HDL Quality Analyzers
        self.quality_analyzer = QualityAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.synthesis_safety_analyzer = SynthesisSafetyAnalyzer()
        self.documentation_analyzer = DocumentationAnalyzer()
        self.maintainability_analyzer = MaintainabilityAnalyzer(
            codebase_path=codebase_path,
            project_root=project_root,
            debug=debug,
        )
        self.verification_coverage_analyzer = VerificationCoverageAnalyzer()

        # HDL-Specific Runtime Analyzers (Initialized with debug flag)
        self.cdc_analyzer = CDCAnalyzer(debug=debug)
        self.uninitialized_signal_analyzer = UninitializedSignalAnalyzer(debug=debug)
        self.signal_integrity_analyzer = SignalIntegrityAnalyzer(debug=debug)

        # HDL deep static analysis adapters (optional — require both flag and availability)
        self.adapters_enabled = enable_adapters and ADAPTERS_AVAILABLE
        if self.adapters_enabled:
            self.unused_module_adapter = UnusedModuleAdapter(debug=debug)
            self.hdl_complexity_adapter = HDLComplexityAdapter(debug=debug)
            self.lint_adapter = LintAdapter(debug=debug)
            self.hierarchy_analyzer_adapter = HierarchyAnalyzerAdapter(debug=debug)
            self.module_metrics_adapter = ModuleMetricsAdapter(debug=debug)
            self.excel_report_adapter = ExcelReportAdapter(
                output_dir=output_dir, debug=debug
            )
            # Dependency graph adapter (HDL dependency analysis)
            if DependencyGraphAdapter is not None:
                self.dependency_graph_adapter = DependencyGraphAdapter(
                    project_root=project_root or codebase_path, debug=debug
                )
            else:
                self.dependency_graph_adapter = None
    
    def _write_metric_report(self, metric_name: str, data: Any) -> None:
        """Write an individual metric report file into the output_dir."""
        filename = f"{metric_name}.json"
        report_path = os.path.join(self.output_dir, filename)

        payload = {
            "metric_name": metric_name,
            "data": data,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    
    def calculate_all_metrics(
        self,
        file_cache: List[Dict[str, Any]],
        hierarchy_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive health metrics for a Verilog/SystemVerilog HDL codebase.
        """
        metrics: Dict[str, Any] = {}

        # 1. Hierarchy/Module Dependency
        metrics["hierarchy_score"] = self._calculate_hierarchy_score(hierarchy_graph)
        self._write_metric_report("hierarchy_score", metrics["hierarchy_score"])

        # 2. Quality
        metrics["quality_score"] = self.quality_analyzer.analyze(file_cache)
        self._write_metric_report("quality_score", metrics["quality_score"])

        # 3. Complexity
        metrics["complexity_score"] = self.complexity_analyzer.analyze(file_cache)
        self._write_metric_report("complexity_score", metrics["complexity_score"])

        # 4. Synthesis Safety (HDL-specific: CDC, uninitialized signals, signal integrity)
        metrics["synthesis_risk_score"] = self._calculate_synthesis_risk_score(file_cache)
        self._write_metric_report("synthesis_risk_score", metrics["synthesis_risk_score"])

        # 5. Lint Score
        metrics["lint_score"] = self.lint_adapter.analyze(file_cache) if self.adapters_enabled else {"score": 100, "grade": "A", "issues": [], "metrics": {}}
        self._write_metric_report("lint_score", metrics["lint_score"])

        # 6. Documentation
        metrics["documentation_score"] = self.documentation_analyzer.analyze(file_cache)
        self._write_metric_report("documentation_score", metrics["documentation_score"])

        # 7. Maintainability
        metrics["maintainability_score"] = self.maintainability_analyzer.analyze(
            file_cache, hierarchy_graph
        )
        self._write_metric_report("maintainability_score", metrics["maintainability_score"])

        # 8. Verification Coverage
        metrics["verification_coverage_score"] = self.verification_coverage_analyzer.analyze(file_cache)
        self._write_metric_report("verification_coverage_score", metrics["verification_coverage_score"])

        # 9. Deep HDL static analysis adapters
        if self.adapters_enabled:
            metrics["adapters"] = self._run_hdl_adapters(file_cache, hierarchy_graph)
            self._write_metric_report("adapter_results", metrics["adapters"])

        # 10. Overall Health
        metrics["overall_health"] = self._calculate_overall_health_score(metrics)

        return metrics

    # ------------------------------------------------------------------ #
    # Deep Static Analysis Adapters
    # ------------------------------------------------------------------ #

    def _init_verible_parser(self) -> Optional[Any]:
        """
        Create a shared Verible parser for HDL analysis.

        Returns None if Verible is not available — adapters
        that require Verible will degrade gracefully.
        """
        try:
            from agents.hdl.verible_parser import VeribleParser

            parser = VeribleParser(debug=self.debug)
            return parser
        except Exception as exc:
            logger.warning(f"Verible parser init failed (adapters will degrade): {exc}")
            return None

    def _run_hdl_adapters(
        self,
        file_cache: List[Dict[str, Any]],
        hierarchy_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute all deep HDL static analysis adapters and generate the Excel report.

        Returns a dict mapping adapter names to their result dicts.
        """
        verible_parser = None
        adapter_results: Dict[str, Any] = {}

        try:
            # Initialize Verible once for all adapters that need it
            verible_parser = self._init_verible_parser()

            # Run each adapter (order: standalone tools first, Verible-dependent second)
            adapters = [
                ("hdl_complexity", self.hdl_complexity_adapter),
                ("lint", self.lint_adapter),
                ("unused_modules", self.unused_module_adapter),
                ("hierarchy", self.hierarchy_analyzer_adapter),
                ("module_metrics", self.module_metrics_adapter),
            ]

            # Add dependency graph adapter if available
            if self.dependency_graph_adapter is not None:
                adapters.append(("dependency_graph", self.dependency_graph_adapter))

            # Single notice when Verible is unavailable
            if verible_parser is None:
                logger.info(
                    "Verible not available — adapters that require it "
                    "(unused_modules, hierarchy, module_metrics) will be skipped."
                )

            for name, adapter in adapters:
                try:
                    result = adapter.analyze(
                        file_cache,
                        verible_parser=verible_parser,
                        hierarchy_graph=hierarchy_graph,
                    )
                    adapter_results[name] = result
                    avail = result.get("tool_available", True)
                    if not avail:
                        logger.info(f"  Adapter {name}: tool not available, skipped")
                    else:
                        score = result.get("score", 0)
                        grade = result.get("grade", "F")
                        logger.info(
                            f"  Adapter {name}: score={score} grade={grade}"
                        )
                except Exception as exc:
                    logger.warning(f"Adapter {name} failed: {exc}")
                    adapter_results[name] = {
                        "score": 0.0,
                        "grade": "F",
                        "metrics": {"error": str(exc)},
                        "issues": [f"Adapter failed: {exc}"],
                        "details": [],
                        "tool_available": False,
                    }

            # Generate detailed_code_review.xlsx from adapter results
            try:
                self.excel_report_adapter.analyze(
                    file_cache=[],
                    adapter_results=adapter_results,
                )
                logger.info("detailed_code_review.xlsx generated successfully")
            except Exception as exc:
                logger.warning(f"Excel report generation failed: {exc}")

        finally:
            # Clean up Verible parser if needed
            if verible_parser is not None:
                try:
                    verible_parser.cleanup()
                except Exception:
                    pass

        return adapter_results

    # ------------------------------------------------------------------ #
    # Synthesis Risk Score (HDL-specific)
    # ------------------------------------------------------------------ #

    def _calculate_synthesis_risk_score(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs comprehensive HDL synthesis risk analyzers and computes a risk score.

        Focuses on:
        - Clock Domain Crossing (CDC) issues
        - Uninitialized signals
        - Signal integrity issues
        """
        # Execute HDL Analyzers
        cdc_res = self.cdc_analyzer.analyze(file_cache)
        uninitialized_res = self.uninitialized_signal_analyzer.analyze(file_cache)
        signal_integrity_res = self.signal_integrity_analyzer.analyze(file_cache)

        # Aggregate Issues
        all_issues = []
        all_issues.extend(cdc_res.get("issues", []))
        all_issues.extend(uninitialized_res.get("issues", []))
        all_issues.extend(signal_integrity_res.get("issues", []))

        # Scoring Logic (Start at 100)
        score = 100.0

        cdc_count = len(cdc_res.get("issues", []))
        uninitialized_count = len(uninitialized_res.get("issues", []))
        signal_integrity_count = len(signal_integrity_res.get("issues", []))

        # CDC issues are critical (high penalty)
        score -= (cdc_count * 20.0)
        # Uninitialized signals are serious
        score -= (uninitialized_count * 15.0)
        # Signal integrity issues have moderate impact
        score -= (signal_integrity_count * 8.0)

        score = max(0.0, score)
        grade = self._score_to_grade(score)

        synthesis_metrics = {
            "cdc_issues": cdc_count,
            "uninitialized_signal_issues": uninitialized_count,
            "signal_integrity_issues": signal_integrity_count,
            "details": {
                "cdc": cdc_res.get("metrics", []),
                "uninitialized": uninitialized_res.get("metrics", []),
                "signal_integrity": signal_integrity_res.get("metrics", [])
            }
        }

        return {
            "score": round(score, 1),
            "grade": grade,
            "issues": all_issues,
            "metrics": synthesis_metrics
        }

    # ------------------------------------------------------------------ #
    # Module Hierarchy Score
    # ------------------------------------------------------------------ #

    def _calculate_hierarchy_score(
        self, hierarchy_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate score based on module hierarchy, instantiation depth, and fanout."""
        if not hierarchy_graph or "analysis" not in hierarchy_graph:
            return {
                "score": 0.0, "grade": "F", "issues": ["No hierarchy graph available"], "metrics": {}
            }

        analysis = hierarchy_graph.get("analysis", {}) or {}
        score = 100.0
        issues: List[str] = []

        # Check for circular instantiations
        cycle_count = analysis.get("cycle_count", 0)
        if cycle_count > 0:
            penalty = min(40.0, float(cycle_count) * 10.0)
            score -= penalty
            issues.append(f"{cycle_count} circular module instantiations detected")

        # Check module fanout (instantiation reuse)
        max_fanout = analysis.get("max_fanout", 0)
        if max_fanout > 20:
            penalty = min(20.0, float(max_fanout - 20) * 1.0)
            score -= penalty
            issues.append(f"High module instantiation depth (max: {max_fanout})")

        # Check for missing module definitions
        missing_count = analysis.get("missing_module_definitions", 0)
        if missing_count > 0:
            penalty = min(15.0, float(missing_count) * 3.0)
            score -= penalty
            issues.append(f"{missing_count} missing module definition(s)")

        # Check module reuse ratio
        total_modules = analysis.get("total_modules", 1)
        unique_modules = analysis.get("unique_modules", 1)
        if total_modules > 0:
            reuse_ratio = unique_modules / total_modules if total_modules > 0 else 1.0
            if reuse_ratio < 0.3:  # Low reuse
                score -= 10.0
                issues.append("Low module reuse ratio detected")

        # Check if hierarchy is balanced
        max_depth = analysis.get("max_instantiation_depth", 0)
        if max_depth > 10:
            penalty = min(10.0, float(max_depth - 10) * 0.5)
            score -= penalty
            issues.append(f"Deep hierarchy detected (max depth: {max_depth})")

        is_connected = analysis.get("is_hierarchy_connected")
        if is_connected is False:
            score -= 5.0
            issues.append("Hierarchy has disconnected module groups")

        score = max(0.0, min(100.0, score))
        grade = self._score_to_grade(score)

        return {
            "score": round(score, 1),
            "grade": grade,
            "issues": issues,
            "metrics": {
                "circular_instantiations": cycle_count,
                "max_fanout": max_fanout,
                "missing_modules": missing_count,
                "total_modules": total_modules,
                "unique_modules": unique_modules,
                "max_depth": max_depth,
            },
        }

    # ------------------------------------------------------------------ #
    # Overall health score
    # ------------------------------------------------------------------ #

    def _calculate_overall_health_score(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall HDL design health from all metric scores."""
        if not metrics:
            return {"score": 0.0, "grade": "F", "recommendation": "No data"}

        # HDL-specific weights (prioritizing synthesis and verification)
        base_weights = {
            "synthesis_risk_score": 0.25,  # Highest: CDC, uninitialized, signal integrity
            "lint_score": 0.20,  # Verilator/Verible lint findings
            "quality_score": 0.15,  # Code quality (style, patterns)
            "maintainability_score": 0.15,  # Module reuse, documentation
            "complexity_score": 0.10,  # Always blocks, module complexity
            "hierarchy_score": 0.05,  # Module hierarchy connectivity
            "documentation_score": 0.05,  # Documentation coverage
            "verification_coverage_score": 0.05,  # Testbench/assertion coverage
        }

        active_weights = {k: w for k, w in base_weights.items() if k in metrics}
        total_w = sum(active_weights.values())

        if total_w <= 0:
            return {"score": 0.0, "grade": "F", "recommendation": "No valid metrics found"}

        weights_used = {k: v / total_w for k, v in active_weights.items()}

        weighted_score = 0.0
        contributions = []

        for name, weight in weights_used.items():
            metric_data = metrics.get(name, {}) or {}
            score = float(metric_data.get("score", 0.0))
            weighted_score += score * weight
            contributions.append({
                "metric": name, "score": score, "weight": weight, "grade": metric_data.get("grade", "F")
            })

        overall_score = weighted_score
        gates_applied = []

        # Critical gates for HDL synthesis
        synth_score = float(metrics.get("synthesis_risk_score", {}).get("score", 100.0))
        if synth_score < 50.0:
            overall_score = min(overall_score, 45.0)
            gates_applied.append("Synthesis Risk Gate: Critical CDC/initialization issues detected.")

        lint_score = float(metrics.get("lint_score", {}).get("score", 100.0))
        if lint_score < 40.0:
            overall_score = min(overall_score, 40.0)
            gates_applied.append("Lint Gate: High lint violation count.")

        qual_score = float(metrics.get("quality_score", {}).get("score", 100.0))
        if qual_score < 40.0:
            overall_score = min(overall_score, 50.0)
            gates_applied.append("Quality Gate: HDL code quality too low.")

        overall_score = round(max(0.0, min(100.0, overall_score)), 1)
        grade = self._score_to_grade(overall_score)

        critical_issues = []
        for name, data in metrics.items():
            if isinstance(data, dict):
                if data.get("score", 100) < 60:
                    critical_issues.extend(data.get("issues", [])[:3])

        deduped_issues = sorted(list(set(critical_issues)))[:10]

        return {
            "score": overall_score,
            "grade": grade,
            "weights_used": weights_used,
            "gates_applied": gates_applied,
            "contributions": contributions,
            "critical_issues": deduped_issues,
            "recommendation": self._get_hdl_health_recommendation(overall_score),
            "pre_gate_score": round(weighted_score, 1),
        }

    def _get_hdl_health_recommendation(self, score: float) -> str:
        """Generate HDL-specific health recommendations."""
        if score >= 90: return "Excellent. Design ready for synthesis."
        if score >= 75: return "Good. Minor refinements recommended before synthesis."
        if score >= 60: return "Acceptable. Check synthesis risks and lint warnings."
        if score >= 40: return "Poor. Address CDC, initialization, and quality issues before synthesis."
        return "Critical. Design unsuitable for synthesis. Severe issues require immediate attention."

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"