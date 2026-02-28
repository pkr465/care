"""
Unused module detection adapter for Verilog/SystemVerilog.

Identifies modules that are not instantiated anywhere in the design hierarchy.
Uses BFS traversal through module instantiation to mark used modules,
treating uninstantiated ones as unused code.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.adapters.base_adapter import BaseStaticAdapter


class UnusedModuleAdapter(BaseStaticAdapter):
    """
    Detects unused code by identifying modules never instantiated.

    Algorithm:
    1. Extract all module definitions from Verilog/SystemVerilog files
    2. Extract all module instantiations
    3. Identify entry points (top-level modules, testbenches)
    4. BFS traversal through instantiation hierarchy to mark reachable modules
    5. Report all uninstantiated modules as unused code
    """

    # Module definition pattern
    _MODULE_DEF_RE = re.compile(r'module\s+(\w+)', re.MULTILINE)

    # Module instantiation pattern: module_type instance_name ( ... )
    _MODULE_INST_RE = re.compile(r'(\w+)\s+(?:#\s*\(.*?\)\s*)?(\w+)\s*\(', re.MULTILINE)

    def __init__(self, debug: bool = False):
        """Initialize unused module adapter."""
        super().__init__("unused_modules", debug=debug)

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        verible_parser: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze files for unused modules.

        Args:
            file_cache: List of file entries with metadata
            verible_parser: Optional Verible parser (unused here)
            dependency_graph: Optional dependency graph (unused here)

        Returns:
            Standard adapter result dict with unused module findings
        """
        # Validation Checks
        if not file_cache:
            return self._create_neutral_result("No files to analyze")

        # Phase 1: Extract all module definitions and instantiations
        all_modules = self._extract_modules(file_cache)

        if not all_modules:
            return self._create_neutral_result("No modules found in analyzed files")

        # Phase 2: Extract module instantiations
        instantiations = self._extract_instantiations(file_cache)

        # Phase 3: Identify entry points (top-level modules, testbenches)
        entry_points = self._identify_entry_points(all_modules, instantiations)

        # Phase 4: BFS reachability analysis through instantiation hierarchy
        reachable = self._perform_reachability_analysis(all_modules, instantiations, entry_points)

        # Phase 5: Report uninstantiated modules
        uninstantiated_keys = set(all_modules.keys()) - reachable

        return self._generate_report(all_modules, entry_points, reachable, uninstantiated_keys)

    def _extract_modules(
        self, file_cache: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[str, int]]:
        """Extracts all module definitions from the provided files."""
        all_modules: Dict[str, Tuple[str, int]] = {}

        for entry in file_cache:
            file_path = entry.get("file_relative_path", entry.get("file_path", "unknown"))
            source_code = entry.get("source", "")

            # Check if this is a Verilog/SystemVerilog file
            if not file_path.endswith((".v", ".sv", ".svh", ".vh")):
                continue

            if not source_code.strip():
                continue

            try:
                # Find all module definitions
                for match in self._MODULE_DEF_RE.finditer(source_code):
                    module_name = match.group(1)
                    start_line = source_code[:match.start()].count('\n') + 1

                    # Create unique key
                    module_key = f"{file_path}::{module_name}"
                    all_modules[module_key] = (file_path, start_line)

            except Exception as e:
                self.logger.warning(
                    f"Error extracting modules from {file_path}: {e}"
                )
                continue

        if self.debug:
            self.logger.debug(f"Extracted {len(all_modules)} modules total.")

        return all_modules

    def _extract_instantiations(
        self, file_cache: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """Extracts all module instantiations (module_type -> instance_name mapping)."""
        instantiations: Dict[str, Set[str]] = {}

        for entry in file_cache:
            file_path = entry.get("file_relative_path", entry.get("file_path", "unknown"))
            source_code = entry.get("source", "")

            if not file_path.endswith((".v", ".sv", ".svh", ".vh")):
                continue

            if not source_code.strip():
                continue

            try:
                # Find all module instantiations
                for match in self._MODULE_INST_RE.finditer(source_code):
                    module_type = match.group(1)
                    # instance_name = match.group(2)

                    # Track which modules are being instantiated
                    if module_type not in instantiations:
                        instantiations[module_type] = set()

            except Exception as e:
                self.logger.warning(
                    f"Error extracting instantiations from {file_path}: {e}"
                )
                continue

        if self.debug:
            self.logger.debug(f"Found {len(instantiations)} module types being instantiated.")

        return instantiations

    def _identify_entry_points(
        self, all_modules: Dict[str, Tuple[str, int]], instantiations: Dict[str, Set[str]]
    ) -> Set[str]:
        """Identifies potential entry points (top-level modules, testbenches)."""
        entry_points: Set[str] = set()

        # Get all module types that are instantiated
        instantiated_types = set(instantiations.keys())

        # Entry points are modules that are NOT instantiated by others
        for module_key, (file_path, _) in all_modules.items():
            module_name = module_key.split("::")[-1]

            # 1. Modules not instantiated anywhere are top-level
            if module_name not in instantiated_types:
                entry_points.add(module_key)
                continue

            # 2. Testbench modules (naming convention)
            if "testbench" in module_name.lower() or module_name.lower().startswith("tb_"):
                entry_points.add(module_key)
                continue

            # 3. Top-level modules in testbench files
            if file_path.endswith("_tb.sv") or file_path.endswith("_tb.v"):
                entry_points.add(module_key)
                continue

        if not entry_points and all_modules:
            self.logger.warning("No entry points found. All modules appear to be instantiated (possible circular hierarchy).")
        elif self.debug:
            self.logger.debug(f"Identified {len(entry_points)} entry points.")

        return entry_points

    def _perform_reachability_analysis(
        self,
        all_modules: Dict[str, Tuple[str, int]],
        instantiations: Dict[str, Set[str]],
        entry_points: Set[str]
    ) -> Set[str]:
        """Performs BFS to find all modules reachable from entry points through instantiation."""
        reachable: Set[str] = set()
        visited: Set[str] = set()
        queue: List[str] = list(entry_points)

        # Pre-populate reachable with entry points
        reachable.update(entry_points)
        visited.update(entry_points)

        processed_count = 0

        while queue:
            current_key = queue.pop(0)
            processed_count += 1

            # Defensive check for very large graphs
            if processed_count > 10000:
                self.logger.warning("Reachability analysis hit safety limit (10000 nodes). Stopping traversal.")
                break

            try:
                current_module_name = current_key.split("::")[-1]

                # Find all modules instantiated by the current module
                # For simplicity, we check if any instantiations match the current module
                instantiated_by_current = instantiations.get(current_module_name, set())

                # For each instantiated child module, find matching entries in all_modules
                for child_module_name in instantiated_by_current:
                    for candidate_key in all_modules:
                        if candidate_key in visited:
                            continue

                        # Check if candidate ends with the child module name
                        if candidate_key.endswith(f"::{child_module_name}"):
                            visited.add(candidate_key)
                            reachable.add(candidate_key)
                            queue.append(candidate_key)

            except Exception as e:
                self.logger.debug(f"Error processing reachability for {current_key}: {e}")
                continue

        return reachable

    def _generate_report(
        self,
        all_modules: Dict[str, Tuple[str, int]],
        entry_points: Set[str],
        reachable: Set[str],
        uninstantiated_keys: Set[str]
    ) -> Dict[str, Any]:
        """Generates the final analysis report."""

        details: List[Dict[str, Any]] = []
        issues: List[str] = []

        # If no unused modules found
        if not uninstantiated_keys:
            return {
                "score": 100.0,
                "grade": "A",
                "metrics": {
                    "total_modules": len(all_modules),
                    "entry_points": len(entry_points),
                    "reachable_count": len(reachable),
                    "unused_count": 0,
                    "unused_percentage": 0.0
                },
                "issues": ["No unused modules detected"],
                "details": [],
                "tool_available": True,
            }

        # Process findings
        for compound_key in sorted(uninstantiated_keys):
            file_path, line = all_modules[compound_key]
            module_name = compound_key.split("::")[-1]

            detail = self._make_detail(
                file=file_path,
                module=module_name,
                line=line,
                description=f"Module '{module_name}' is never instantiated",
                severity="medium",
                category="unused_module",
                drc="",
            )
            details.append(detail)

        issues.append(
            f"Found {len(uninstantiated_keys)} unused modules (never instantiated)"
        )

        # Calculate score: Start at 100, deduct points.
        unused_count = len(uninstantiated_keys)
        total_modules = len(all_modules)

        # Penalty: 2 points per unused module, capped at 0.
        score = max(0.0, 100.0 - (unused_count * 2.0))

        grade = self._score_to_grade(score)

        metrics = {
            "total_modules": total_modules,
            "entry_points": len(entry_points),
            "reachable_count": len(reachable),
            "unused_count": unused_count,
            "unused_percentage": (unused_count / total_modules * 100) if total_modules > 0 else 0.0,
        }

        return {
            "score": score,
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _create_neutral_result(self, message: str) -> Dict[str, Any]:
        """Returns a neutral (passing) result when no analysis is possible/needed."""
        return {
            "score": 100.0,
            "grade": "A",
            "metrics": {},
            "issues": [message],
            "details": [],
            "tool_available": True
        }