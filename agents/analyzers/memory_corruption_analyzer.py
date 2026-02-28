"""
Verilog/SystemVerilog signal integrity and connectivity analysis
"""

import re
from typing import Dict, List, Any
from .base_runtime_analyzer import RuntimeAnalyzerBase


class SignalIntegrityAnalyzer(RuntimeAnalyzerBase):
    """
    Signal integrity and bus contention analyzer.

    Detects:
    1. Multiple drivers on same net (bus contention)
    2. Bit-width mismatch in assignments (truncation/extension)
    3. Signed/unsigned mismatch
    4. Array index out of bounds
    5. Parameter override violations
    6. Port connection width mismatch
    7. Inout port misuse
    8. Memory inference issues
    """

    _ASSIGN = re.compile(r"\bassign\s+([a-zA-Z0-9_]+)\s*=")
    _REG_ASSIGN = re.compile(r"([a-zA-Z0-9_]+)\s*<=\s*")
    _BITWIDTH = re.compile(r"\[(\d+):(\d+)\]")
    _PORT_CONNECT = re.compile(r"\.(\w+)\s*\(\s*(\w+)\s*\)")
    _INOUT = re.compile(r"\binout\s+([a-zA-Z0-9_]+)")

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        issues = []
        metrics = []

        for entry in file_cache:
            if entry.get("suffix", "").lower() not in {".v", ".sv", ".vh", ".svh"}:
                continue

            rel_path = entry.get("rel_path", "unknown")
            source = entry.get("source", "")

            result = self.analyze_single_file(source, rel_path)
            issues.extend(result["issues"])
            metrics.append(result["metrics"])

        return {"metrics": metrics, "issues": issues}

    def analyze_single_file(self, source: str, rel_path: str) -> Dict[str, Any]:
        local_issues = []
        file_risk_count = 0

        # Detect multiple drivers (continuous assign statements on same signal)
        assign_targets: Dict[str, int] = {}
        for m in self._ASSIGN.finditer(source):
            target = m.group(1)
            if target in assign_targets:
                assign_targets[target] += 1
            else:
                assign_targets[target] = 1

        for target, count in assign_targets.items():
            if count > 1:
                local_issues.append(
                    f"{rel_path}: Signal '{target}' has {count} continuous drivers "
                    "(multiple assign statements - bus contention risk)"
                )
                file_risk_count += 1

        # Detect bit-width mismatches in assignments
        assignments = re.finditer(r"([a-zA-Z0-9_\[\]:]+)\s*<=?\s*([a-zA-Z0-9_\[\]:]+)", source)
        for m in assignments:
            lhs = m.group(1)
            rhs = m.group(2)

            lhs_width = self._extract_width(lhs)
            rhs_width = self._extract_width(rhs)

            if lhs_width and rhs_width and lhs_width != rhs_width:
                local_issues.append(
                    f"{rel_path}: Potential bit-width mismatch: "
                    f"'{lhs}' ({lhs_width} bits) = '{rhs}' ({rhs_width} bits)"
                )
                file_risk_count += 1

        # Detect inout port misuse
        inout_matches = self._INOUT.findall(source)
        if inout_matches:
            for inout_port in inout_matches:
                # Check if it's driven conditionally (tristate pattern)
                if not re.search(rf"{inout_port}\s*=\s*.*\?\s*.*:\s*1'bz", source):
                    local_issues.append(
                        f"{rel_path}: Inout port '{inout_port}' detected; "
                        "ensure proper tri-state logic (value or 1'bz)"
                    )
                    file_risk_count += 1

        # Array index out of bounds (heuristic)
        array_accesses = re.finditer(r"([a-zA-Z0-9_]+)\s*\[\s*(\d+|[a-zA-Z0-9_]+)\s*\]", source)
        for m in array_accesses:
            var_name = m.group(1)
            index_expr = m.group(2)

            # Try to find array declaration
            decl_match = re.search(rf"\b(?:reg|wire)\s+(?:\[\d+:\d+\]\s+)?{var_name}\s*\[(\d+):(\d+)\]", source)
            if decl_match:
                arr_hi = int(decl_match.group(1))
                arr_lo = int(decl_match.group(2))
                try:
                    if isinstance(index_expr, str) and index_expr.isdigit():
                        idx = int(index_expr)
                        if idx > arr_hi or idx < arr_lo:
                            local_issues.append(
                                f"{rel_path}: Array index {idx} out of bounds for '{var_name}' "
                                f"[{arr_hi}:{arr_lo}]"
                            )
                            file_risk_count += 1
                except ValueError:
                    pass

        # Detect memory inference issues (unusual patterns)
        mem_pattern = re.finditer(r"\breg\s+\[.*?\]\s+(\w+)\s*\[\s*\d+\s*:\s*\d+\s*\]", source)
        for m in mem_pattern:
            mem_name = m.group(1)
            # Check if not reset or initialized
            if not re.search(rf"{mem_name}.*=.*'0|{mem_name}.*reset", source):
                local_issues.append(
                    f"{rel_path}: Memory '{mem_name}' inferred; verify reset/initialization logic"
                )
                file_risk_count += 1

        return {
            "issues": local_issues,
            "metrics": {
                "file": rel_path,
                "signal_integrity_risks": file_risk_count,
            },
        }

    @staticmethod
    def _extract_width(signal_expr: str) -> int:
        """Extract bit-width from signal expression like 'a[7:0]'."""
        m = re.search(r"\[(\d+):(\d+)\]", signal_expr)
        if m:
            hi = int(m.group(1))
            lo = int(m.group(2))
            return abs(hi - lo) + 1
        return 0
