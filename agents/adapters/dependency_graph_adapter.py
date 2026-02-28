"""
Dependency Graph Adapter for CARE HDL Analysis Framework.

Wraps HDLDependencyAnalyzer for the MetricsCalculator pipeline, providing
a standardized interface to perform comprehensive dependency analysis on
Verilog/SystemVerilog projects.

The adapter scores the dependency graph across multiple dimensions:
- Module hierarchy and instantiation relationships
- Include dependencies and circular includes
- Package imports and symbol resolution
- Parameter propagation and type checking
- Interface bindings and modports
- Generate block expansion

Scoring starts at 100 and deducts points for various quality issues.
"""

import logging
from typing import Any, Dict, List, Optional

from agents.adapters.base_adapter import BaseStaticAdapter
from agents.services import DependencyGraph
from agents.analyzers.dependency_analyzer import HDLDependencyAnalyzer, AnalyzerConfig

logger = logging.getLogger("adapters.dependency_graph")


class DependencyGraphAdapter(BaseStaticAdapter):
    """
    Adapter for HDL dependency analysis using HDLDependencyAnalyzer.

    Analyzes module hierarchies, include dependencies, package imports,
    parameter propagation, interface bindings, and symbol resolution.
    Produces a scored dependency graph suitable for design health metrics.

    Args:
        project_root: Root directory of the HDL project.
        debug: Enable debug logging if True.
        **kwargs: Additional configuration options passed to AnalyzerConfig:
            - ignore_dirs: Directories to skip during analysis
            - include_paths: Additional paths to search for includes
            - include_extensions: File extensions to analyze (default: ['.v', '.sv'])
            - max_include_depth: Maximum include nesting depth (default: 20)
            - detect_circular_includes: Enable circular include detection (default: True)
            - resolve_packages: Attempt to resolve package imports (default: True)
            - detect_symbol_collisions: Enable symbol collision detection (default: True)
            - expand_generate_blocks: Expand generate block instantiations (default: True)
    """

    def __init__(self, project_root: str, debug: bool = False, **kwargs):
        """Initialize the dependency graph adapter."""
        super().__init__("dependency_graph", debug=debug)
        self.project_root = project_root

        # Build AnalyzerConfig from kwargs
        config_params = {
            "project_root": project_root,
            "ignore_dirs": kwargs.get("ignore_dirs", []),
            "include_paths": kwargs.get("include_paths", []),
            "max_include_depth": kwargs.get("max_include_depth", 2),
            "max_hierarchy_depth": kwargs.get("max_hierarchy_depth", 10),
            "use_verible": kwargs.get("use_verible", True),
            "verible_timeout": kwargs.get("verible_timeout", 30),
            "exclude_system_packages": kwargs.get("exclude_system_packages", True),
            "track_parameters": kwargs.get("track_parameters", True),
            "track_interfaces": kwargs.get("track_interfaces", True),
            "debug": debug,
        }

        try:
            self.config = AnalyzerConfig(**config_params)
            self.analyzer = HDLDependencyAnalyzer(self.config)
            self._last_graph: Optional[DependencyGraph] = None
            self.logger.info(
                f"Initialized DependencyGraphAdapter for {project_root}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize HDLDependencyAnalyzer: {e}")
            raise

    # ── Public API ───────────────────────────────────────────────────────

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        verible_parser: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run dependency analysis and return standardized result dict.

        Args:
            file_cache: List of file dicts with 'file_path' and 'source' keys.
            verible_parser: Optional Verible parser (not used by this adapter).
            dependency_graph: Optional pre-computed dependency graph (not used).

        Returns:
            Standard adapter result dict with score, grade, metrics, issues, details.
        """
        try:
            if not file_cache:
                return self._empty_result("No files to analyze")

            # Run the analyzer
            graph = self.analyzer.analyze(file_cache)
            self._last_graph = graph

            # Score the graph
            score = self._score_graph(graph)
            grade = self._score_to_grade(score)

            # Build metrics from the graph summary
            metrics = self._build_metrics(graph)
            metrics["score"] = score

            # Collect issues and details
            issues = self._collect_issues(graph)
            details = self._collect_details(graph)

            result = {
                "score": score,
                "grade": grade,
                "metrics": metrics,
                "issues": issues,
                "details": details,
                "tool_available": True,
            }

            self.logger.info(
                f"Dependency analysis complete: {score:.0f} ({grade}) "
                f"| {graph.total_modules} modules, {graph.total_includes} includes"
            )
            return result

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}", exc_info=True)
            return {
                "score": 0.0,
                "grade": "F",
                "metrics": {"tool_available": False, "error": str(e)},
                "issues": [f"Analysis error: {e}"],
                "details": [],
                "tool_available": False,
            }

    def get_dependency_graph(self) -> Optional[DependencyGraph]:
        """Return the last computed DependencyGraph for reuse by other adapters."""
        return self._last_graph

    # ── Scoring Logic ────────────────────────────────────────────────────

    def _score_graph(self, graph: DependencyGraph) -> float:
        """Score the dependency graph; start at 100 and deduct for issues."""
        score = 100.0

        # Circular module dependencies: -10 each (max -40)
        cycles = len(graph.module_hierarchy.cycles)
        score -= min(cycles * 10, 40)

        # Circular includes: -5 each (max -20)
        circular_includes = len(graph.include_tree.circular_includes)
        score -= min(circular_includes * 5, 20)

        # Unresolved includes: -2 each (max -10)
        unresolved_includes = len(graph.include_tree.unresolved_includes)
        score -= min(unresolved_includes * 2, 10)

        # High fan-out modules (out_degree > 15): -3 each (max -15)
        high_fanout = sum(
            1 for m in graph.module_hierarchy.modules.values()
            if m.fan_out > 15
        )
        score -= min(high_fanout * 3, 15)

        # Symbol collisions: -2 each (max -10)
        symbol_collisions = graph.symbol_table.total_collisions
        score -= min(symbol_collisions * 2, 10)

        # Parameter type mismatches: -3 each (max -15)
        param_mismatches = graph.parameter_map.total_mismatches
        score -= min(param_mismatches * 3, 15)

        # Orphan modules (defined but never instantiated, excluding testbenches): -1 each (max -10)
        orphans = self._count_orphan_modules(graph)
        score -= min(orphans * 1, 10)

        # Hierarchy depth > 10: -1 per extra level (max -5)
        if graph.module_hierarchy.max_depth > 10:
            depth_penalty = min(graph.module_hierarchy.max_depth - 10, 5)
            score -= depth_penalty

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def _count_orphan_modules(self, graph: DependencyGraph) -> int:
        """Count modules that are defined but never instantiated (excluding testbenches)."""
        instantiated = set()
        for inst in graph.module_hierarchy.instantiations:
            instantiated.add(inst.child_module)

        orphans = 0
        for module_name, module in graph.module_hierarchy.modules.items():
            if (
                module_name not in instantiated
                and not module.is_testbench
                and module_name not in graph.module_hierarchy.root_modules
            ):
                orphans += 1

        return orphans

    # ── Metrics Building ───────────────────────────────────────────────────

    def _build_metrics(self, graph: DependencyGraph) -> Dict[str, Any]:
        """Extract metrics from the dependency graph."""
        summary = graph.score_summary
        return {
            "tool_available": True,
            "modules": summary.get("modules", 0),
            "instantiations": summary.get("instantiations", 0),
            "max_hierarchy_depth": summary.get("max_depth", 0),
            "module_cycles": summary.get("cycles", 0),
            "includes": summary.get("includes", 0),
            "unresolved_includes": summary.get("unresolved_includes", 0),
            "circular_includes": summary.get("circular_includes", 0),
            "packages": summary.get("packages", 0),
            "package_imports": summary.get("imports", 0),
            "unresolved_packages": summary.get("unresolved_packages", 0),
            "parameter_overrides": summary.get("parameter_overrides", 0),
            "parameter_type_mismatches": summary.get("param_mismatches", 0),
            "interfaces": summary.get("interfaces", 0),
            "interface_bindings": summary.get("interface_bindings", 0),
            "generate_blocks": summary.get("generate_blocks", 0),
            "symbols_resolved": summary.get("symbols", 0),
            "symbol_collisions": summary.get("symbol_collisions", 0),
        }

    # ── Issue Collection ───────────────────────────────────────────────────

    def _collect_issues(self, graph: DependencyGraph) -> List[str]:
        """Generate human-readable issue summaries."""
        issues = []

        # Circular module dependencies
        if graph.module_hierarchy.cycles:
            issues.append(
                f"{len(graph.module_hierarchy.cycles)} module cycle(s) detected; "
                "circular dependencies complicate testing and synthesis"
            )

        # Circular includes
        if graph.include_tree.circular_includes:
            issues.append(
                f"{len(graph.include_tree.circular_includes)} circular include(s); "
                "may cause compilation failures or unexpected behavior"
            )

        # Unresolved includes
        if graph.include_tree.unresolved_includes:
            unresolved_count = len(graph.include_tree.unresolved_includes)
            issues.append(
                f"{unresolved_count} unresolved include(s); "
                "verify include paths and file naming"
            )

        # High fan-out modules
        high_fanout = [
            m for m in graph.module_hierarchy.modules.values() if m.fan_out > 15
        ]
        if high_fanout:
            issues.append(
                f"{len(high_fanout)} module(s) with high fan-out (>15); "
                "consider hierarchical restructuring"
            )

        # Symbol collisions
        if graph.symbol_table.total_collisions > 0:
            issues.append(
                f"{graph.symbol_table.total_collisions} symbol collision(s); "
                "may cause ambiguous resolution or unexpected overrides"
            )

        # Parameter type mismatches
        if graph.parameter_map.total_mismatches > 0:
            issues.append(
                f"{graph.parameter_map.total_mismatches} parameter type mismatch(es); "
                "verify parameter override types match module definitions"
            )

        # Orphan modules
        orphans = self._count_orphan_modules(graph)
        if orphans > 0:
            issues.append(
                f"{orphans} orphan module(s) defined but never instantiated; "
                "remove unused code or verify instantiation"
            )

        # Deep hierarchy
        if graph.module_hierarchy.max_depth > 10:
            issues.append(
                f"Hierarchy depth {graph.module_hierarchy.max_depth} exceeds 10; "
                "consider flattening or re-organizing module structure"
            )

        return issues

    # ── Details Collection ───────────────────────────────────────────────

    def _collect_details(self, graph: DependencyGraph) -> List[Dict[str, Any]]:
        """Generate detailed finding records for Excel export."""
        details = []

        # Module cycles
        for cycle_idx, cycle in enumerate(graph.module_hierarchy.cycles):
            cycle_str = " -> ".join(cycle)
            details.append(
                self._make_detail(
                    file="",
                    module=cycle[0] if cycle else "unknown",
                    line=0,
                    description=f"Circular module dependency: {cycle_str}",
                    severity="critical",
                    category="module_cycles",
                    drc="DEP-001",
                )
            )

        # Circular includes
        for src, dst in graph.include_tree.circular_includes:
            details.append(
                self._make_detail(
                    file=src,
                    module="",
                    line=0,
                    description=f"Circular include: {src} <-> {dst}",
                    severity="high",
                    category="include_cycles",
                    drc="DEP-002",
                )
            )

        # Unresolved includes
        for unresolved in graph.include_tree.unresolved_includes:
            details.append(
                self._make_detail(
                    file=unresolved.source_file,
                    module="",
                    line=unresolved.line,
                    description=f"Unresolved include: {unresolved.include_name}",
                    severity="high",
                    category="unresolved_includes",
                    drc="DEP-003",
                )
            )

        # High fan-out modules
        for module in sorted(
            graph.module_hierarchy.modules.values(),
            key=lambda m: -m.fan_out,
        ):
            if module.fan_out > 15:
                details.append(
                    self._make_detail(
                        file=module.file_path,
                        module=module.name,
                        line=module.line,
                        description=f"High fan-out: {module.fan_out} children instantiated",
                        severity="medium",
                        category="high_fanout",
                        drc="DEP-004",
                    )
                )

        # Symbol collisions
        for symbol_name, symbol_defs in graph.symbol_table.collisions:
            details.append(
                self._make_detail(
                    file=symbol_defs[0].file_path if symbol_defs else "",
                    module=symbol_name,
                    line=symbol_defs[0].line if symbol_defs else 0,
                    description=f"Symbol collision: {symbol_name} defined in multiple files",
                    severity="medium",
                    category="symbol_collision",
                    drc="DEP-005",
                )
            )

        # Parameter type mismatches
        for mismatch in graph.parameter_map.type_mismatches:
            details.append(
                self._make_detail(
                    file=mismatch.file_path,
                    module=mismatch.instance_name,
                    line=mismatch.line,
                    description=(
                        f"Parameter type mismatch: {mismatch.param_name} "
                        f"({mismatch.param_type}); override={mismatch.override_value} "
                        f"default={mismatch.default_value}"
                    ),
                    severity="high",
                    category="param_mismatch",
                    drc="DEP-006",
                )
            )

        # Orphan modules
        instantiated = {inst.child_module for inst in graph.module_hierarchy.instantiations}
        for module_name, module in graph.module_hierarchy.modules.items():
            if (
                module_name not in instantiated
                and not module.is_testbench
                and module_name not in graph.module_hierarchy.root_modules
            ):
                details.append(
                    self._make_detail(
                        file=module.file_path,
                        module=module_name,
                        line=module.line,
                        description=f"Orphan module: defined but never instantiated",
                        severity="low",
                        category="orphan_module",
                        drc="DEP-007",
                    )
                )

        return details
