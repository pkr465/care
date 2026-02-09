"""
Dead code detection adapter using CCLS code navigator.

Identifies functions that are not reachable from known entry points via call graph analysis.
Uses BFS traversal to mark reachable functions, treating unreachable ones as potential dead code.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.adapters.base_adapter import BaseStaticAdapter


class DeadCodeAdapter(BaseStaticAdapter):
    """
    Detects dead code by identifying functions unreachable from entry points.

    Algorithm:
    1. Extract all function definitions from C/C++ files
    2. Identify entry points (main, test functions, header exports)
    3. BFS traversal from entry points to mark reachable functions
    4. Report all unreachable functions as dead code
    """

    def __init__(self, debug: bool = False):
        """Initialize dead code adapter."""
        super().__init__("dead_code", debug=debug)

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        ccls_navigator: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze files for dead code.

        Args:
            file_cache: List of file entries with metadata
            ccls_navigator: CCLSCodeNavigator instance for code analysis
            dependency_graph: Optional dependency graph (unused here)

        Returns:
            Standard adapter result dict with dead code findings
        """
        # Validation checks
        if not file_cache:
            return self._empty_result("No files to analyze")

        if ccls_navigator is None:
            return self._handle_tool_unavailable(
                "CCLSCodeNavigator",
                "Dead code analysis requires CCLS; ensure ccls is installed and indexed",
            )

        # Phase 1: Extract all function definitions
        all_functions: Dict[str, Tuple[str, int]] = {}
        for entry in file_cache:
            language = entry.get("language", "").lower()
            if language not in ("c", "cpp"):
                continue

            file_path = entry.get("file_path") or entry.get("file_name")
            if not file_path:
                continue

            try:
                doc = ccls_navigator.create_doc(file_path)
                if doc is None:
                    self.logger.debug(f"Could not create doc for {file_path}")
                    continue

                ccls_navigator.openDoc(doc)
                symbols_dict = ccls_navigator.getDocumentSymbolsKeySymbols(doc)

                if not symbols_dict:
                    continue

                # Extract functions from symbols
                rel_path = entry.get("file_relative_path", file_path)
                for func_name, symbol_list in symbols_dict.items():
                    for symbol in symbol_list:
                        if symbol.get("kind") == "Function":
                            location = symbol.get("location", {})
                            range_info = location.get("range", {})
                            start_info = range_info.get("start", {})
                            line = start_info.get("line", 0)

                            compound_key = f"{rel_path}::{func_name}"
                            all_functions[compound_key] = (file_path, line)
                            break  # Use first Function occurrence

            except Exception as e:
                self.logger.warning(
                    f"Error extracting functions from {file_path}: {e}"
                )
                continue

        if not all_functions:
            return self._empty_result("No functions found in analyzed files")

        # Phase 2: Identify entry points
        entry_points: Set[str] = set()
        known_entries = {"main", "_start", "__libc_start_main", "WinMain", "DllMain"}

        for compound_key, (file_path, line) in all_functions.items():
            # Extract function name from compound key
            func_name = compound_key.split("::")[-1]

            # Check if it's a known entry point
            if func_name in known_entries:
                entry_points.add(compound_key)
                continue

            # Check if it's a test function
            if func_name.lower().startswith("test"):
                entry_points.add(compound_key)
                continue

            # Check if it's defined in a header file (likely API export)
            if file_path.endswith((".h", ".hpp")):
                entry_points.add(compound_key)
                continue

        if not entry_points:
            self.logger.debug(
                "No entry points found; treating all functions as potentially dead"
            )

        # Phase 3: BFS reachability analysis
        reachable: Set[str] = set()
        visited: Set[str] = set()
        queue: List[str] = list(entry_points)

        while queue and len(visited) < 500:  # Limit to prevent hangs
            current_key = queue.pop(0)
            if current_key in visited:
                continue

            visited.add(current_key)
            reachable.add(current_key)

            # Get callees from current function
            try:
                file_path, line = all_functions[current_key]
                doc = ccls_navigator.create_doc(file_path)
                if doc is None:
                    continue

                ccls_navigator.openDoc(doc)
                func_name = current_key.split("::")[-1]

                # Try to find symbol and get its position
                symbols_dict = ccls_navigator.getDocumentSymbolsKeySymbols(doc)
                if func_name not in symbols_dict:
                    continue

                symbol_list = symbols_dict[func_name]
                if not symbol_list:
                    continue

                symbol = symbol_list[0]
                doc, pos = ccls_navigator.getDocandPosFromSymbol(symbol)
                if doc is None or pos is None:
                    continue

                # Get callees
                callee_tree = ccls_navigator.getCallee(doc, pos, level=2)
                callee_names = self._extract_names_from_tree(callee_tree)

                # Add new callees to queue
                for callee_name in callee_names:
                    # Try compound key matching first
                    for key in all_functions.keys():
                        if key.endswith(f"::{callee_name}") and key not in visited:
                            queue.append(key)
                            break
                    else:
                        # Fall back to simple name matching
                        for key in all_functions.keys():
                            if key.split("::")[-1] == callee_name and key not in visited:
                                queue.append(key)
                                break

            except Exception as e:
                self.logger.debug(f"Error processing reachability for {current_key}: {e}")
                continue

        # Phase 4: Report unreachable functions
        unreachable_keys = set(all_functions.keys()) - reachable
        details: List[Dict[str, Any]] = []
        issues: List[str] = []

        for compound_key in sorted(unreachable_keys):
            file_path, line = all_functions[compound_key]
            func_name = compound_key.split("::")[-1]

            detail = self._make_detail(
                file=file_path,
                function=func_name,
                line=line,
                description=f"Function '{func_name}' is unreachable from entry points",
                severity="medium",
                category="dead_code",
            )
            details.append(detail)

        if unreachable_keys:
            issues.append(
                f"Found {len(unreachable_keys)} unreachable functions (potential dead code)"
            )

        # Calculate score: 100 - (3 points per dead function)
        dead_count = len(unreachable_keys)
        score = max(0, 100 - dead_count * 3)

        # Compute metrics
        metrics = {
            "total_functions": len(all_functions),
            "entry_points": len(entry_points),
            "reachable_count": len(reachable),
            "dead_count": dead_count,
            "dead_percentage": (
                (dead_count / len(all_functions) * 100)
                if all_functions
                else 0.0
            ),
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
