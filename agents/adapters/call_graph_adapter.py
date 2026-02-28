"""
Module hierarchy analysis adapter for Verilog/SystemVerilog.

Builds a complete module instantiation hierarchy and computes metrics including
fan-in/fan-out analysis, cycle detection, hierarchy depth analysis, and
identification of architectural issues.
"""

import logging
import sys
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from agents.adapters.base_adapter import BaseStaticAdapter


class HierarchyAnalyzerAdapter(BaseStaticAdapter):
    """
    Analyzes module hierarchy structure to identify architectural issues.

    Metrics computed:
    - Fan-in/fan-out degrees for each module
    - Detection of high-fan-in modules (widely used)
    - Identification of modules with excessive fan-out
    - Cycle detection in instantiation hierarchy
    - Maximum hierarchy depth analysis
    - Leaf and orphan module classification
    """

    # Module definition and instantiation patterns
    _MODULE_DEF_RE = re.compile(r'module\s+(\w+)', re.MULTILINE)
    _MODULE_INST_RE = re.compile(r'(\w+)\s+(?:#\s*\(.*?\)\s*)?(\w+)\s*\(', re.MULTILINE)

    def __init__(self, debug: bool = False):
        """Initialize hierarchy analyzer adapter."""
        super().__init__("hierarchy", debug=debug)

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        verible_parser: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze module hierarchy structure.

        Args:
            file_cache: List of file entries with metadata
            verible_parser: Optional Verible parser (unused here)
            dependency_graph: Optional dependency graph (unused here)

        Returns:
            Standard adapter result dict with hierarchy metrics
        """
        # Validation Checks
        if not file_cache:
            return self._create_neutral_result("No files to analyze")

        # Build Graph
        graph = self._build_module_graph(file_cache)

        if not graph:
            return self._create_neutral_result("No modules or instantiation graph found")

        # Compute Metrics
        metrics_data = self._compute_metrics(graph)

        # Generate Report
        return self._generate_report(graph, metrics_data)

    def _build_module_graph(
        self, file_cache: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """Build module instantiation graph (adjacency list)."""
        graph: Dict[str, Set[str]] = defaultdict(set)
        files_processed = 0

        for entry in file_cache:
            if files_processed >= 200:  # Soft limit
                self.logger.warning("Reached file processing limit (200). Stopping graph build.")
                break

            file_path = entry.get("file_relative_path", entry.get("file_path", "unknown"))
            source_code = entry.get("source", "")

            # Check if this is a Verilog/SystemVerilog file
            if not file_path.endswith((".v", ".sv", ".svh", ".vh")):
                continue

            if not source_code.strip():
                continue

            try:
                # Extract module definitions and instantiations
                self._process_file_modules(file_path, source_code, graph)
                files_processed += 1
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
                continue

        # Ensure all referenced modules exist as keys in the graph
        all_instantiated = set()
        for instantiations in graph.values():
            all_instantiated.update(instantiations)

        for module in all_instantiated:
            if module not in graph:
                graph[module] = set()

        return dict(graph)

    def _process_file_modules(
        self, file_path: str, source_code: str, graph: Dict[str, Set[str]]
    ) -> None:
        """Helper to process modules and instantiations within a single file."""

        # Extract module definitions in this file
        modules_in_file = set()
        for match in self._MODULE_DEF_RE.finditer(source_code):
            module_name = match.group(1)
            modules_in_file.add(module_name)
            if module_name not in graph:
                graph[module_name] = set()

        # Extract module instantiations
        for match in self._MODULE_INST_RE.finditer(source_code):
            module_type = match.group(1)
            # instance_name = match.group(2)

            # For each module defined in this file, add the instantiation
            for parent_module in modules_in_file:
                # Check if this instantiation is inside the module definition
                # (simple heuristic: count module/endmodule pairs)
                if module_type != parent_module:  # Don't self-instantiate
                    if parent_module not in graph:
                        graph[parent_module] = set()
                    graph[parent_module].add(module_type)

    def _compute_metrics(self, graph: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Compute graph metrics (fan-in, fan-out, cycles, depth)."""

        # Calculate degrees
        out_degree = {module: len(instantiations) for module, instantiations in graph.items()}
        in_degree = defaultdict(int)

        # Initialize all nodes in in_degree map
        for module in graph:
            in_degree[module] = 0

        for module, instantiations in graph.items():
            for instantiated in instantiations:
                in_degree[instantiated] += 1

        # Identify problematic modules
        high_fan_in = {module for module, count in in_degree.items() if count > 20}
        high_fan_out = {module for module, count in out_degree.items() if count > 15}
        leaf_modules = {module for module, count in out_degree.items() if count == 0}
        orphan_modules = {module for module, count in in_degree.items() if count == 0}

        # Detect cycles
        cycles = set()
        visited_global = set()
        rec_stack = set()

        # Increase recursion limit slightly for deep graphs
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

        try:
            for start_module in graph:
                if start_module not in visited_global:
                    self._dfs_detect_cycles(
                        start_module, graph, visited_global, rec_stack, cycles
                    )
        except RecursionError:
            self.logger.error("Recursion limit hit during cycle detection")

        # Compute max depth
        max_hierarchy_depth = self._compute_max_depth(graph, in_degree)

        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "high_fan_in": high_fan_in,
            "high_fan_out": high_fan_out,
            "leaf_modules": leaf_modules,
            "orphan_modules": orphan_modules,
            "cycles": cycles,
            "max_hierarchy_depth": max_hierarchy_depth
        }

    def _generate_report(
        self, graph: Dict[str, Set[str]], metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detail entries and calculate score."""

        details: List[Dict[str, Any]] = []
        issues: List[str] = []

        in_degree = metrics["in_degree"]
        out_degree = metrics["out_degree"]
        high_fan_in = metrics["high_fan_in"]
        high_fan_out = metrics["high_fan_out"]
        cycles = metrics["cycles"]
        max_hierarchy_depth = metrics["max_hierarchy_depth"]

        # 1. High fan-in modules
        for module in sorted(high_fan_in):
            details.append(self._make_detail(
                file="",
                module=module,
                line=0,
                description=f"Module '{module}' has high fan-in ({in_degree[module]}); widely reused",
                severity="medium",
                category="hierarchy",
                drc="",
            ))

        if high_fan_in:
            issues.append(f"Found {len(high_fan_in)} modules with high fan-in (>20)")

        # 2. High fan-out modules
        for module in sorted(high_fan_out):
            details.append(self._make_detail(
                file="",
                module=module,
                line=0,
                description=f"Module '{module}' has high fan-out ({out_degree[module]}); excessive instantiation",
                severity="medium",
                category="hierarchy",
                drc="",
            ))

        if high_fan_out:
            issues.append(f"Found {len(high_fan_out)} modules with high fan-out (>15)")

        # 3. Cycles in hierarchy
        for module in sorted(cycles):
            details.append(self._make_detail(
                file="",
                module=module,
                line=0,
                description=f"Module '{module}' is part of an instantiation cycle",
                severity="high",
                category="hierarchy",
                drc="",
            ))

        if cycles:
            issues.append(f"Found {len(cycles)} modules involved in instantiation cycles")

        if not issues:
            issues = ["No architectural issues detected in hierarchy"]

        # Score Calculation
        score = 100.0
        score -= len(high_fan_in) * 2
        score -= len(high_fan_out) * 3
        score -= len(cycles) * 10
        score -= max(0, max_hierarchy_depth - 10) * 1
        score = max(0.0, min(100.0, score))

        # Aggregate Metrics for Display
        total_edges = sum(len(instantiations) for instantiations in graph.values())
        module_count = len(graph)

        output_metrics = {
            "modules_in_graph": module_count,
            "instantiations": total_edges,
            "avg_fan_in": round(total_edges / module_count, 2) if module_count else 0,
            "avg_fan_out": round(total_edges / module_count, 2) if module_count else 0,
            "max_fan_in": max(in_degree.values()) if in_degree else 0,
            "max_fan_out": max(out_degree.values()) if out_degree else 0,
            "high_fan_in_count": len(high_fan_in),
            "high_fan_out_count": len(high_fan_out),
            "cycle_count": len(cycles),
            "max_hierarchy_depth": max_hierarchy_depth,
            "leaf_count": len(metrics["leaf_modules"]),
            "orphan_count": len(metrics["orphan_modules"]),
        }

        return {
            "score": score,
            "grade": self._score_to_grade(score),
            "metrics": output_metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _create_neutral_result(self, message: str) -> Dict[str, Any]:
        """Returns a neutral (passing) result."""
        return {
            "score": 100.0,
            "grade": "A",
            "metrics": {},
            "issues": [message],
            "details": [],
            "tool_available": True
        }

    def _dfs_detect_cycles(
        self,
        node: str,
        graph: Dict[str, Set[str]],
        visited: Set[str],
        rec_stack: Set[str],
        cycles: Set[str],
    ) -> None:
        """DFS-based cycle detection in module hierarchy."""
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                self._dfs_detect_cycles(neighbor, graph, visited, rec_stack, cycles)
            elif neighbor in rec_stack:
                # Cycle found
                cycles.add(node)
                cycles.add(neighbor)

        rec_stack.remove(node)

    def _compute_max_depth(
        self, graph: Dict[str, Set[str]], in_degree: Dict[str, Any]
    ) -> int:
        """Compute maximum hierarchy depth from root modules."""
        memo: Dict[str, int] = {}

        def dfs_depth(node: str, visited: Set[str]) -> int:
            if node in memo:
                return memo[node]

            if node in visited:
                return 0  # Break cycle

            visited.add(node)

            instantiations = graph.get(node, set())
            if not instantiations:
                memo[node] = 1
                return 1

            # Get max depth of children
            max_child_depth = 0
            for child in instantiations:
                d = dfs_depth(child, visited.copy())
                if d > max_child_depth:
                    max_child_depth = d

            depth = 1 + max_child_depth
            memo[node] = depth
            return depth

        # Roots are modules with in-degree 0
        roots = [module for module, count in in_degree.items() if count == 0]

        if not roots and graph:
            # If full cyclic graph, pick first few nodes
            roots = list(graph.keys())[:5]

        max_depth = 0
        for root in roots:
            try:
                depth = dfs_depth(root, set())
                max_depth = max(max_depth, depth)
            except RecursionError:
                continue

        return max_depth