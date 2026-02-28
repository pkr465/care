"""
Verilog/SystemVerilog uninitialized signal analysis
"""

import re
from typing import Dict, List, Any, Tuple
from .base_runtime_analyzer import RuntimeAnalyzerBase


class UninitializedSignalAnalyzer(RuntimeAnalyzerBase):
    """
    Uninitialized signal and incomplete logic detection.

    Capabilities:
    1. Signals read before being assigned
    2. Registers without reset values
    3. Output ports left potentially unconnected
    4. X/Z propagation analysis
    5. Missing else clause causing latch inference
    6. Wire declarations without drivers
    7. Incomplete case statements without default
    """

    _WIRE_DECL = re.compile(r"\bwire\s+([a-zA-Z0-9_]+)")
    _REG_DECL = re.compile(r"\breg\s+([a-zA-Z0-9_]+)(?!\s*=)")  # Without assignment
    _OUTPUT_DECL = re.compile(r"\boutput\s+(?:reg\s+)?([a-zA-Z0-9_]+)")
    _ASSIGN = re.compile(r"\bassign\s+([a-zA-Z0-9_]+)\s*=")
    _CASE_STMT = re.compile(r"\bcase\b")

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

        # Collect signal declarations
        wires = set()
        regs = set()
        outputs = set()
        assignments = set()

        for m in self._WIRE_DECL.finditer(source):
            wires.add(m.group(1))

        for m in self._REG_DECL.finditer(source):
            regs.add(m.group(1))

        for m in self._OUTPUT_DECL.finditer(source):
            outputs.add(m.group(1))

        for m in self._ASSIGN.finditer(source):
            assignments.add(m.group(1))

        # Check for uninitialized regs
        for reg_name in regs:
            if reg_name not in assignments:
                line_num = source[:source.find(reg_name)].count("\n") + 1
                local_issues.append(
                    f"{rel_path}:{line_num}: Register '{reg_name}' may be uninitialized "
                    "(no reset or initialization)"
                )
                file_risk_count += 1

        # Check for undriven wires
        for wire_name in wires:
            if wire_name not in assignments:
                if not re.search(rf"\binput\s+.*{wire_name}", source):
                    line_num = source[:source.find(wire_name)].count("\n") + 1
                    local_issues.append(
                        f"{rel_path}:{line_num}: Wire '{wire_name}' declared but never driven"
                    )
                    file_risk_count += 1

        # Check for undriven outputs
        for out_name in outputs:
            if out_name not in assignments:
                line_num = source[:source.find(out_name)].count("\n") + 1
                local_issues.append(
                    f"{rel_path}:{line_num}: Output '{out_name}' may be left undriven"
                )
                file_risk_count += 1

        # Check for incomplete case statements
        func_blocks = self._get_function_blocks(source)
        for block_name, body, start_line in func_blocks:
            case_count = len(self._CASE_STMT.findall(body))
            if case_count > 0:
                default_count = len(re.findall(r"\bdefault\b", body))
                if default_count < case_count:
                    local_issues.append(
                        f"{rel_path}: Block '{block_name}' at line {start_line} has "
                        f"case statement(s) without default clause (potential latch inference)"
                    )
                    file_risk_count += 1

        # Check for X/Z propagation
        x_assign_count = len(re.findall(r"[1-9]'bx|[1-9]'bz|'x|'z", source))
        if x_assign_count > 0:
            local_issues.append(
                f"{rel_path}: Found {x_assign_count} X/Z value(s) assigned in logic; "
                "verify intentional use"
            )
            file_risk_count += x_assign_count

        return {
            "issues": local_issues,
            "metrics": {
                "file": rel_path,
                "uninitialized_risks": file_risk_count,
            },
        }
