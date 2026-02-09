"""
Call graph analysis adapter using CCLS code navigator.

Builds a complete call graph and computes metrics including fan-in/fan-out analysis,
cycle detection, call depth analysis, and identification of architectural issues.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.adapters.base_adapter import BaseStaticAdapter


class CallGraphAdapter(BaseStaticAdapter):
    """
    Analyzes call graph structure to identify architectural issues.

    Metrics computed:
    - Fan-in/fan-out degrees for each function
    - Detection of high-coupling God functions
    - Identification of functions with excessive responsibilities
    - Cycle detection in call graph
    - Maximum call depth analysis
    - Orphan and leaf function classification
    """

    def __init__(self, debug: bool = False):
        """Initialize call graph adapter."""
        super().__init__("call_graph", debug=debug)

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        ccls_navigator: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze call graph structure.

        Args:
            file_cache: List of file entries with metadata
            ccls_navigator: CCLSCodeNavigator instance for code analysis
            dependency_graph: Optional dependency graph (unused here)

        Returns:
            Standard adapter result dict with call graph metrics
        """
        # Validation checks
        if not file_cache or ccls_navigator is None:
            return self._handle_tool_unavailable(
                "CCLSCodeNavigator",
                "Call graph analysis requires CCLS; ensure ccls is installed and indexed",
            )

        # Phase 1: Build call graph adjacency list
        graph: Dict[str, Set[str]] = {}
        file_count = 0
        function_count = 0

        for entry in file_cache:
            if file_count >= 200:  # Limit files processed
                break

            language = entry.get("language", "").lower()
            if language not in ("c", "cpp"):
                continue

            file_path = entry.get("file_path") or entry.get("file_name")
            if not file_path:
                continue

            try:
                doc = ccls_navigator.create_doc(file_path)
                if doc is None:
                    continue

                ccls_navigator.openDoc(doc)
                symbols_dict = ccls_navigator.getDocumentSymbolsKeySymbols(doc)

                if not symbols_dict:
                    continue

                functions_in_file = 0
                for func_name, symbol_list in symbols_dict.items():
                    if functions_in_file >= 50:  # Limit functions per file
                        break

                    for symbol in symbol_list:
                        if symbol.get("kind") == "Function":
                            if func_name not in graph:
                                graph[func_name] = set()

                            # Get callees for this function
                            try:
                                doc_result, pos = ccls_navigator.getDocandPosFromSymbol(
                                    symbol
                                )
                                if doc_result is None or pos is None:
                                    break

                                callee_tree = ccls_navigator.getCallee(doc_result, pos, level=1)
                                callee_names = self._extract_names_from_tree(callee_tree)

                                for callee_name in callee_names:
                                    if callee_name != func_name:  # Avoid self-loops
                                        graph[func_name].add(callee_name)

                            except Exception as e:
                                self.logger.debug(
                                    f"Error getting callees for {func_name}: {e}"
                                )

                            functions_in_file += 1
                            function_count += 1
                            break

                file_count += 1

            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
                continue

        if not graph:
            return self._empty_result("No functions or call graph found")

        # Ensure all referenced functions are in graph
        all_callees = set()
        for callees in graph.values():
            all_callees.update(callees)
        for callee in all_callees:
            if callee not in graph:
                graph[callee] = set()

        # Phase 2: Compute metrics
        in_degree: Dict[str, int] = {func: 0 for func in graph}
        out_degree: Dict[str, int] = {func: len(callees) for func, callees in graph.items()}

        # Calculate in-degree
        for func, callees in graph.items():
            for callee in callees:
                in_degree[callee] += 1

        # Identify problematic functions
        high_fan_in: Set[str] = {func for func in graph if in_degree[func] > 20}
        high_fan_out: Set[str] = {func for func in graph if out_degree[func] > 15}
        leaf_functions: Set[str] = {func for func in graph if out_degree[func] == 0}
        orphan_functions: Set[str] = {func for func in graph if in_degree[func] == 0}

        # Phase 3: Detect cycles
        cycles: Set[str] = set()
        for start_func in graph:
            visited: Set[str] = set()
            rec_stack: Set[str] = set()
            self._dfs_detect_cycles(start_func, graph, visited, rec_stack, cycles)

        # Phase 4: Compute maximum call depth
        max_call_depth = self._compute_max_depth(graph, in_degree)

        # Phase 5: Create detail entries
        details: List[Dict[str, Any]] = []
        issues: List[str] = []

        # High fan-in functions
        for func in sorted(high_fan_in):
            detail = self._make_detail(
                file="",
                function=func,
                line=0,
                description=(
                    f"Function '{func}' has high fan-in ({in_degree[func]} callers); "
                    "potential God function or tight coupling"
                ),
                severity="medium",
                category="call_graph",
            )
            details.append(detail)

        if high_fan_in:
            issues.append(f"Found {len(high_fan_in)} functions with high fan-in (>20)")

        # High fan-out functions
        for func in sorted(high_fan_out):
            detail = self._make_detail(
                file="",
                function=func,
                line=0,
                description=(
                    f"Function '{func}' has high fan-out ({out_degree[func]} callees); "
                    "doing too much, consider refactoring"
                ),
                severity="medium",
                category="call_graph",
            )
            details.append(detail)

        if high_fan_out:
            issues.append(f"Found {len(high_fan_out)} functions with high fan-out (>15)")

        # Cyclic functions
        for func in sorted(cycles):
            detail = self._make_detail(
                file="",
                function=func,
                line=0,
                description=f"Function '{func}' is part of a circular call cycle",
                severity="high",
                category="call_graph",
            )
            details.append(detail)

        if cycles:
            issues.append(f"Found {len(cycles)} functions involved in call cycles")

        # Phase 6: Calculate score
        score = 100.0
        score -= len(high_fan_in) * 3
        score -= len(high_fan_out) * 4
        score -= len(cycles) * 10
        score -= max(0, max_call_depth - 10) * 2
        score = max(0, min(100, score))

        # Compute aggregate metrics
        total_edges = sum(len(callees) for callees in graph.values())
        avg_fan_in = total_edges / len(graph) if graph else 0
        avg_fan_out = total_edges / len(graph) if graph else 0
        max_fan_in = max(in_degree.values()) if in_degree else 0
        max_fan_out = max(out_degree.values()) if out_degree else 0

        metrics = {
            "functions_in_graph": len(graph),
            "edges": total_edges,
            "avg_fan_in": round(avg_fan_in, 2),
            "avg_fan_out": round(avg_fan_out, 2),
            "max_fan_in": max_fan_in,
            "max_fan_out": max_fan_out,
            "high_fan_in_count": len(high_fan_in),
            "high_fan_out_count": len(high_fan_out),
            "cycle_count": len(cycles),
            "max_call_depth": max_call_depth,
            "leaf_count": len(leaf_functions),
            "orphan_count": len(orphan_functions),
        }

        return {
            "score": score,
            "grade": self._score_to_grade(score),
            "metrics": metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _extract_names_from_tree(
        self, tree: Optional[Dict[str, Any]], collected: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract function names from call tree structure.

        Recursively walks the tree structure returned by getCallee,
        collecting all function names.

        Args:
            tree: Call tree dict with 'name' and 'children' keys
            collected: Accumulated list of names (for recursion)

        Returns:
            List of function names found in tree
        """
        if collected is None:
            collected = []

        if tree is None or not isinstance(tree, dict):
            return collected

        # Add current node's name if present
        if "name" in tree:
            name = tree["name"]
            if name and name not in collected:
                collected.append(name)

        # Recurse on children
        children = tree.get("children", [])
        if isinstance(children, list):
            for child in children:
                self._extract_names_from_tree(child, collected)

        return collected

    def _dfs_detect_cycles(
        self,
        node: str,
        graph: Dict[str, Set[str]],
        visited: Set[str],
        rec_stack: Set[str],
        cycles: Set[str],
    ) -> None:
        """
        DFS-based cycle detection using recursion stack.

        Marks any function involved in a cycle. Uses standard DFS algorithm
        where a back edge (neighbor in rec_stack) indicates a cycle.

        Args:
            node: Current function being visited
            graph: Adjacency list representation of call graph
            visited: Set of all visited nodes
            rec_stack: Set of nodes in current recursion path
            cycles: Accumulator set for functions in cycles
        """
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                self._dfs_detect_cycles(neighbor, graph, visited, rec_stack, cycles)
            elif neighbor in rec_stack:
                # Back edge found: cycle detected
                cycles.add(node)
                cycles.add(neighbor)

        rec_stack.remove(node)

    def _compute_max_depth(
        self, graph: Dict[str, Set[str]], in_degree: Dict[str, int]
    ) -> int:
        """
        Compute maximum call depth in the graph.

        Uses memoized DFS from all root functions (in_degree == 0).
        Handles cycles by using a visited set per traversal to prevent infinite loops.

        Args:
            graph: Adjacency list call graph
            in_degree: Dictionary mapping functions to in-degree

        Returns:
            Maximum depth found, or 0 if no roots exist
        """
        memo: Dict[str, int] = {}

        def dfs_depth(node: str, visited: Set[str]) -> int:
            """Compute depth from a single node, handling cycles."""
            if node in memo:
                return memo[node]

            if node in visited:
                return 0  # Cycle: stop here

            visited.add(node)

            callees = graph.get(node, set())
            if not callees:
                memo[node] = 1
                return 1

            max_child_depth = max(
                (dfs_depth(callee, visited.copy()) for callee in callees),
                default=0,
            )
            depth = 1 + max_child_depth
            memo[node] = depth
            return depth

        # Find all root functions (in_degree == 0)
        roots = [func for func in graph if in_degree[func] == 0]

        if not roots:
            return 0

        max_depth = 0
        for root in roots:
            depth = dfs_depth(root, set())
            max_depth = max(max_depth, depth)

        return max_depth
