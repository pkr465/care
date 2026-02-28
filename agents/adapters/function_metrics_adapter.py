"""
Module Metrics Adapter for Verilog/SystemVerilog code analysis.

Analyzes module-level metrics including port counts, always block types,
generate blocks, parameters, and module size to assess design quality.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from agents.adapters.base_adapter import BaseStaticAdapter


class ModuleMetricsAdapter(BaseStaticAdapter):
    """
    Analyzes module-level metrics in Verilog/SystemVerilog code.

    Detects:
    - Modules with excessive ports (>30, >80)
    - Large modules (>300, >800 lines)
    - Latch inference issues
    - Multiple always block types per module
    - Generate block usage
    - Parameterized modules
    """

    # Module definition pattern
    _MODULE_DEF_RE = re.compile(r'module\s+(\w+).*?\n(.*?)\bendmodule\b', re.MULTILINE | re.DOTALL)
    # Port detection in module header
    _PORT_RE = re.compile(r'\([\s\S]*?\)', re.MULTILINE)
    # Always block detection
    _ALWAYS_RE = re.compile(r'always\s*@', re.MULTILINE)
    _ALWAYS_FF_RE = re.compile(r'always_ff\s*@', re.MULTILINE)
    _ALWAYS_COMB_RE = re.compile(r'always_comb', re.MULTILINE)
    _ALWAYS_LATCH_RE = re.compile(r'always_latch', re.MULTILINE)
    # Generate block detection
    _GENERATE_RE = re.compile(r'generate\b', re.MULTILINE)
    # Parameter detection
    _PARAMETER_RE = re.compile(r'parameter\s+', re.MULTILINE)

    def __init__(self, debug: bool = False):
        """Initialize ModuleMetricsAdapter."""
        super().__init__("module_metrics", debug)

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        verible_parser: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze modules in Verilog/SystemVerilog files for design metrics and issues.

        Args:
            file_cache: List of file entries with language, path, etc.
            verible_parser: Optional Verible parser (unused here).
            dependency_graph: Unused for this adapter.

        Returns:
            Standard result dict with score, grade, metrics, issues, details.
        """
        # Validation
        if not file_cache:
            return self._create_neutral_result("No files to analyze")

        all_modules = []

        for file_entry in file_cache:
            file_path = file_entry.get("file_relative_path", file_entry.get("file_path", "unknown"))
            source_code = file_entry.get("source", "")

            # Check if this is a Verilog/SystemVerilog file
            if not file_path.endswith((".v", ".sv", ".svh", ".vh")):
                continue

            if not source_code.strip():
                continue

            try:
                # Extract module metrics
                modules = self._extract_module_info(file_path, source_code)
                if modules:
                    all_modules.extend(modules)

            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")

        # Check results
        if not all_modules:
            return self._create_neutral_result("No modules found in Verilog/SystemVerilog files")

        # Compute aggregates and collect details
        metrics, issues, details = self._compute_metrics(all_modules)

        # Calculate score
        score = self._calculate_score(metrics)
        grade = self._score_to_grade(score)

        return {
            "score": score,
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _extract_module_info(
        self, file_path: str, source_code: str
    ) -> List[Dict[str, Any]]:
        """Extract module metrics from Verilog/SystemVerilog source."""
        modules = []

        for match in self._MODULE_DEF_RE.finditer(source_code):
            module_name = match.group(1)
            module_body = match.group(2)
            start_line = source_code[:match.start()].count('\n') + 1
            end_line = source_code[:match.end()].count('\n') + 1
            body_lines = end_line - start_line

            # Count ports
            port_match = self._PORT_RE.search(match.group(0))
            port_count = 0
            if port_match:
                port_str = port_match.group(0)
                port_count = port_str.count(',') + (1 if port_str.strip() != '()' else 0)

            # Count always block types
            always_count = len(self._ALWAYS_RE.findall(module_body))
            always_ff_count = len(self._ALWAYS_FF_RE.findall(module_body))
            always_comb_count = len(self._ALWAYS_COMB_RE.findall(module_body))
            always_latch_count = len(self._ALWAYS_LATCH_RE.findall(module_body))

            # Count generate blocks
            generate_count = len(self._GENERATE_RE.findall(module_body))

            # Count parameters
            param_count = len(self._PARAMETER_RE.findall(module_body))

            modules.append({
                "file": file_path,
                "name": module_name,
                "line": start_line,
                "body_lines": body_lines,
                "port_count": port_count,
                "always_blocks": always_count,
                "always_ff_blocks": always_ff_count,
                "always_comb_blocks": always_comb_count,
                "always_latch_blocks": always_latch_count,
                "generate_blocks": generate_count,
                "parameter_count": param_count,
            })

        return modules


    def _compute_metrics(
        self, all_modules: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """Compute aggregate metrics and identify issues."""

        metrics = {
            "total_modules": len(all_modules),
            "total_always_blocks": sum(m.get("always_blocks", 0) for m in all_modules),
            "total_always_ff": sum(m.get("always_ff_blocks", 0) for m in all_modules),
            "total_always_comb": sum(m.get("always_comb_blocks", 0) for m in all_modules),
            "total_always_latch": sum(m.get("always_latch_blocks", 0) for m in all_modules),
            "total_generate_blocks": sum(m.get("generate_blocks", 0) for m in all_modules),
            "modules_with_many_ports": 0,
            "modules_with_excessive_ports": 0,
            "avg_ports": 0.0,
            "max_ports": 0,
            "large_modules": 0,
            "very_large_modules": 0,
            "avg_body_lines": 0.0,
            "max_body_lines": 0,
            "parameterized_module_count": 0,
        }

        issues = []
        details = []

        total_ports = 0
        total_body_lines = 0
        parameterized_count = 0

        for module in all_modules:
            # Ports
            port_count = module["port_count"]
            total_ports += port_count
            metrics["max_ports"] = max(metrics["max_ports"], port_count)

            if port_count > 30:
                metrics["modules_with_many_ports"] += 1
            if port_count > 80:
                metrics["modules_with_excessive_ports"] += 1

            # Body lines
            body_lines = module["body_lines"]
            total_body_lines += body_lines
            metrics["max_body_lines"] = max(metrics["max_body_lines"], body_lines)

            if body_lines > 300:
                metrics["large_modules"] += 1
            if body_lines > 800:
                metrics["very_large_modules"] += 1

            # Parameterized modules
            if module["parameter_count"] > 0:
                parameterized_count += 1

            # Latch inference penalty
            latch_count = module.get("always_latch_blocks", 0)

            # Generate detail entries for issues
            if port_count > 30:
                severity = "high" if port_count > 80 else "medium"
                detail = self._make_detail(
                    file=module["file"],
                    module=module["name"],
                    line=module["line"],
                    description=f"Module has {port_count} ports (high complexity)",
                    severity=severity,
                    category="module_design",
                    drc="",
                )
                details.append(detail)
                issues.append(f"{module['name']} has {port_count} ports")

            # Large modules
            if body_lines > 300:
                severity = "high" if body_lines > 800 else "medium"
                detail = self._make_detail(
                    file=module["file"],
                    module=module["name"],
                    line=module["line"],
                    description=f"Module body is {body_lines} lines (exceeds ideal size)",
                    severity=severity,
                    category="module_design",
                    drc="",
                )
                details.append(detail)
                issues.append(f"{module['name']} body is {body_lines} lines long")

            # Latch warnings
            if latch_count > 0:
                detail = self._make_detail(
                    file=module["file"],
                    module=module["name"],
                    line=module["line"],
                    description=f"Module has {latch_count} always_latch blocks (potential latches)",
                    severity="high",
                    category="module_design",
                    drc="HDL-003",
                )
                details.append(detail)
                issues.append(f"{module['name']} may infer latches (always_latch)")

        # Calculate averages
        if all_modules:
            metrics["avg_ports"] = round(total_ports / len(all_modules), 2)
            metrics["avg_body_lines"] = round(total_body_lines / len(all_modules), 2)

        metrics["parameterized_module_count"] = parameterized_count

        return metrics, issues, details

    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall score based on module metrics.

        Penalty: -1 per module with many ports, -2 per excessive ports,
                 -1 per large module, -3 per very large,
                 -5 per latch block.
        """
        score = 100.0
        score -= metrics["modules_with_many_ports"]
        score -= 2 * metrics["modules_with_excessive_ports"]
        score -= metrics["large_modules"]
        score -= 3 * metrics["very_large_modules"]
        score -= metrics["total_always_latch"] * 5

        # Clamp to [0, 100]
        return max(0.0, min(100.0, score))

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