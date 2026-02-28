"""
Verilog/SystemVerilog synthesis safety and design rule checking
"""

import os
import re
from typing import Dict, List, Any
from collections import Counter
import math


class SynthesisSafetyAnalyzer:
    """
    Analyzes Verilog/SystemVerilog code for synthesis safety and design rule violations:
    - Combinational loops (output feeding back to input)
    - Latch inference in combinational logic
    - Clock domain crossing (CDC) without synchronization
    - Metastability risks (single-stage synchronizers)
    - X-propagation paths
    - Uninitialized registers (no reset value)
    - Race conditions from blocking assignments
    - Tri-state buses in FPGA designs
    - Async reset without sync de-assertion
    - Clock gating without proper cell usage
    Uses custom HDL-DRC (Design Rule Check) codes: HDL-DRC-001, HDL-DRC-002, etc.
    """

    # Verilog/SystemVerilog file extensions
    V_EXTS = {".v", ".sv"}
    VH_EXTS = {".vh", ".svh"}

    def __init__(self, codebase_path: str = None, project_root: str = None):
        """Initialize synthesis safety analyzer."""
        self.codebase_path = codebase_path or os.getcwd()
        self.project_root = project_root or os.getcwd()
        self._file_cache: List[Dict[str, Any]] = []

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog for synthesis safety violations.

        Args:
            file_cache: List of processed Verilog file entries

        Returns:
            Synthesis safety analysis results with score, grade, metrics, issues
        """
        self._file_cache = file_cache or []
        return self._calculate_safety_score()

    def _calculate_safety_score(self) -> Dict[str, Any]:
        """
        Synthesis safety analysis for Verilog/SystemVerilog codebases.

        Scans for HDL-specific design rule violations and synthesis hazards.
        """
        if not self._file_cache:
            return {
                "score": 0.0,
                "grade": "F",
                "issues": ["No files cached"],
                "metrics": {
                    "files_analyzed": 0,
                    "total_violations": 0,
                    "risk_points": 0,
                    "critical_rule_present": False,
                    "top_violation_types": [],
                    "violations_by_file": {},
                    "severity_breakdown": {},
                    "rule_counts": {},
                    "files_with_violations": 0,
                    "clean_files": 0,
                    "files_with_critical": 0,
                    "files_with_high": 0,
                },
            }

        def _rel_path(entry: Dict[str, Any]) -> str:
            p = (
                entry.get("rel_path")
                or entry.get("path")
                or entry.get("file_relative_path")
                or entry.get("file_name")
                or ""
            )
            root = (
                getattr(self, "project_root", None)
                or getattr(self, "root_dir", None)
                or str(self.codebase_path)
            )
            try:
                return os.path.relpath(p, root) if p else ""
            except Exception:
                return p or ""

        # Filter Verilog files
        all_exts = {ext.lower() for ext in (self.V_EXTS | self.VH_EXTS)}
        v_files: List[Dict[str, Any]] = []
        for f in self._file_cache:
            suffix = (f.get("suffix") or "").lower()
            if suffix in all_exts:
                v_files.append(f)

        print(
            f"DEBUG synthesis_safety: Found {len(v_files)} Verilog files out of "
            f"{len(self._file_cache)} total"
        )

        if not v_files:
            return {
                "score": 100.0,
                "grade": "A",
                "metrics": {
                    "files_analyzed": 0,
                    "total_violations": 0,
                    "risk_points": 0,
                    "critical_rule_present": False,
                    "top_violation_types": [],
                    "violations_by_file": {},
                    "severity_breakdown": {},
                    "rule_counts": {},
                    "files_with_violations": 0,
                    "clean_files": 0,
                    "files_with_critical": 0,
                    "files_with_high": 0,
                },
                "issues": [
                    "No Verilog/SystemVerilog files found for synthesis safety analysis. "
                    f"Extensions checked: {sorted(self.V_EXTS | self.VH_EXTS)}"
                ],
            }

        # Severity weights for scoring
        severity_weight = {"critical": 10, "high": 6, "medium": 3, "low": 1}

        # Rule catalog for HDL design rule checks
        rules = [
            # Combinational logic hazards
            {
                "id": "HDL-DRC-001",
                "sev": "critical",
                "category": "combinational_loop",
                "desc": "Potential combinational loop detected",
                "pat": r"assign\s+\w+.*?;\s*assign\s+\w+.*?\1",
                "fix": "Break feedback loops with registers or verify feedback is intentional",
            },
            {
                "id": "HDL-DRC-002",
                "sev": "high",
                "category": "latch_inference",
                "desc": "Potential latch inference in combinational always block",
                "pat": r"always\s*@\*.*?(?:if|case).*?(?!else).*?end",
                "fix": "Ensure all paths in combinational logic assign all outputs",
            },
            {
                "id": "HDL-DRC-003",
                "sev": "high",
                "category": "incomplete_sensitivity",
                "desc": "Incomplete sensitivity list in always block",
                "pat": r"always\s*@\s*\([^)]*\)(?!.*always\s*@\s*\*)",
                "fix": "Use always @* for combinational logic or include all inputs",
            },

            # Clock domain crossing
            {
                "id": "HDL-DRC-004",
                "sev": "critical",
                "category": "cdc",
                "desc": "Clock domain crossing without synchronizer detected",
                "pat": r"(posedge|negedge)\s+(\w+).*?(posedge|negedge)\s+(?!\2)",
                "fix": "Add CDC synchronizer (gray code or multi-stage FF) between clock domains",
            },
            {
                "id": "HDL-DRC-005",
                "sev": "high",
                "category": "metastability",
                "desc": "Potential single-stage synchronizer (metastability risk)",
                "pat": r"reg\s+\w+\s*=\s*(?:input|wire).*?reg\s+\w+\s*=\s*\1",
                "fix": "Use at least 2-stage flip-flop synchronizer for CDC",
            },

            # Initialization and reset
            {
                "id": "HDL-DRC-006",
                "sev": "medium",
                "category": "uninitialized",
                "desc": "Register may be uninitialized (no reset value)",
                "pat": r"reg\s+\w+\s*;(?!.*reset)",
                "fix": "Initialize register in reset condition or at declaration",
            },
            {
                "id": "HDL-DRC-007",
                "sev": "high",
                "category": "async_reset",
                "desc": "Async reset without sync de-assertion detected",
                "pat": r"if\s*\(\s*!?reset\s*\)(?!.*negedge.*reset)",
                "fix": "Synchronize async reset de-assertion using proper reset sequencing",
            },

            # Blocking vs non-blocking
            {
                "id": "HDL-DRC-008",
                "sev": "high",
                "category": "race_condition",
                "desc": "Blocking assignment in sequential always block (race condition risk)",
                "pat": r"always\s*@\s*\(posedge.*?[^<]\s*=\s*",
                "fix": "Use non-blocking (<=) assignments in sequential always blocks",
            },

            # Tri-state and FPGA issues
            {
                "id": "HDL-DRC-009",
                "sev": "high",
                "category": "tristate",
                "desc": "Tri-state output in FPGA (may not synthesize optimally)",
                "pat": r"assign\s+\w+\s*=\s*.*?\s*\?\s*.*?\s*:\s*1'bz",
                "fix": "Use regular logic or confirm tri-state is required for ASIC",
            },

            # X-propagation
            {
                "id": "HDL-DRC-010",
                "sev": "medium",
                "category": "x_propagation",
                "desc": "X-propagation path detected (undefined value handled)",
                "pat": r"=\s*[1-9]'bx|=\s*'x|=\s*\{.*?'x.*?\}",
                "fix": "Avoid X in logic; use proper reset and initialization",
            },

            # Clock gating
            {
                "id": "HDL-DRC-011",
                "sev": "medium",
                "category": "clock_gating",
                "desc": "Clock gating without ICG cell (may cause glitches)",
                "pat": r"assign\s+clk_out\s*=\s*clk.*?&.*?enable",
                "fix": "Use dedicated integrated clock gating (ICG) cell for safe clock gating",
            },

            # Multi-bit CDC
            {
                "id": "HDL-DRC-012",
                "sev": "high",
                "category": "multibit_cdc",
                "desc": "Multi-bit signal crossing clock domain without gray code",
                "pat": r"always\s*@\s*\(posedge.*?\[.*?:.*?\].*?posedge\s+(?!.*gray)",
                "fix": "Use gray code or handshake protocol for multi-bit CDC",
            },

            # Async FIFO
            {
                "id": "HDL-DRC-013",
                "sev": "medium",
                "category": "async_fifo",
                "desc": "Async FIFO detected - verify pointer synchronization",
                "pat": r"(?:async_fifo|fifo.*clock|read_clk.*write_clk)",
                "fix": "Ensure read/write pointers synchronized across clock domains",
            },
        ]

        def _strip_comments_keep_strings(src: str) -> str:
            out: List[str] = []
            i, n = 0, len(src)
            in_sl = in_ml = in_str = in_chr = False
            esc = False

            while i < n:
                ch = src[i]
                nxt = src[i + 1] if i + 1 < n else ""

                if in_sl:
                    if ch == "\n":
                        in_sl = False
                        out.append("\n")
                    i += 1
                    continue

                if in_ml:
                    if ch == "*" and nxt == "/":
                        in_ml = False
                        i += 2
                        continue
                    if ch == "\n":
                        out.append("\n")
                    i += 1
                    continue

                if in_str:
                    out.append(ch)
                    if not esc and ch == '"':
                        in_str = False
                    esc = (ch == "\\" and not esc)
                    i += 1
                    continue

                if in_chr:
                    out.append(ch)
                    if not esc and ch == "'":
                        in_chr = False
                    esc = (ch == "\\" and not esc)
                    i += 1
                    continue

                if ch == "/" and nxt == "/":
                    in_sl = True
                    i += 2
                    continue

                if ch == "/" and nxt == "*":
                    in_ml = True
                    i += 2
                    continue

                if ch == '"':
                    in_str = True
                    esc = False
                    out.append(ch)
                    i += 1
                    continue

                if ch == "'":
                    in_chr = True
                    esc = False
                    out.append(ch)
                    i += 1
                    continue

                out.append(ch)
                i += 1

            return "".join(out)

        # Pre-compile rules
        compiled_rules = []
        for r in rules:
            flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
            compiled = {**r, "regex": re.compile(r["pat"], flags)}
            compiled_rules.append(compiled)

        print(
            f"DEBUG synthesis_safety: Starting analysis of {len(v_files)} files "
            f"with {len(compiled_rules)} rules"
        )

        files_analyzed = 0
        violations_by_file: Dict[str, List[Dict[str, Any]]] = {}
        rule_counts: Counter = Counter()

        severity_breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        files_with_critical = 0
        files_with_high = 0

        # Scan files
        for entry in v_files:
            source = entry.get("source") or ""
            if not source.strip():
                print(
                    f"DEBUG synthesis_safety: Skipping empty file: "
                    f"{entry.get('file_relative_path', 'unknown')}"
                )
                continue

            files_analyzed += 1
            rel = _rel_path(entry)

            code = _strip_comments_keep_strings(source)
            code_lines = code.splitlines()
            print(
                f"DEBUG synthesis_safety: Analyzing {rel} "
                f"({len(source)} chars, {len(code)} chars after comment stripping)"
            )

            file_hits: List[Dict[str, Any]] = []
            file_rule_counts: Counter = Counter()
            file_severity_seen = {"critical": False, "high": False}

            for r in compiled_rules:
                matches = list(r["regex"].finditer(code))
                if not matches:
                    continue

                for m in matches:
                    start = m.start()
                    line_idx = code.count("\n", 0, start)
                    line = line_idx + 1
                    last_nl = code.rfind("\n", 0, start)
                    col = (start - (last_nl + 1)) if last_nl >= 0 else start

                    line_text = code_lines[line_idx] if 0 <= line_idx < len(code_lines) else ""
                    snippet = line_text.strip()[:200]

                    hit = {
                        "rule": r["id"],
                        "severity": r["sev"],
                        "category": r.get("category", ""),
                        "description": r["desc"],
                        "line": line,
                        "column": col + 1,
                        "snippet": snippet,
                        "remediation": r.get("fix", ""),
                    }
                    file_hits.append(hit)
                    rule_counts[r["id"]] += 1
                    file_rule_counts[r["id"]] += 1
                    severity_breakdown[r["sev"]] = severity_breakdown.get(r["sev"], 0) + 1

                    if r["sev"] == "critical":
                        file_severity_seen["critical"] = True
                    if r["sev"] == "high":
                        file_severity_seen["high"] = True

            if file_hits:
                violations_by_file[rel] = file_hits
                if file_severity_seen["critical"]:
                    files_with_critical += 1
                if file_severity_seen["high"]:
                    files_with_high += 1
                print(
                    f"DEBUG synthesis_safety: File {rel} - {len(file_hits)} violations: "
                    f"{dict(file_rule_counts)}"
                )
            else:
                print(f"DEBUG synthesis_safety: File {rel} - No violations found")

        total_violations = sum(rule_counts.values())
        print(
            f"DEBUG synthesis_safety: Analysis complete - {files_analyzed} files analyzed, "
            f"{total_violations} total violations"
        )

        # Compute risk score
        risk_points = 0
        critical_present = False

        for rel, hits in violations_by_file.items():
            for h in hits:
                sev = h["severity"]
                w = severity_weight.get(sev, 1)
                risk_points += w
                if sev == "critical":
                    critical_present = True

        print(
            f"DEBUG synthesis_safety: Risk points: {risk_points}, "
            f"Critical present: {critical_present}"
        )
        print(f"DEBUG synthesis_safety: Severity breakdown: {severity_breakdown}")

        # Score model
        if files_analyzed > 0:
            normalization = 1.0 / max(1.0, math.sqrt(float(files_analyzed)))
        else:
            normalization = 1.0

        normalized_risk = risk_points * normalization
        score = 100.0 - min(90.0, normalized_risk * 2.0)
        if critical_present:
            score = min(score, 50.0)
        score = max(0.0, score)

        grade = self._score_to_grade(score)
        print(f"DEBUG synthesis_safety: Final score: {score:.1f}")

        files_with_violations = len(violations_by_file)
        clean_files = files_analyzed - files_with_violations

        # Top violation types
        top_types = sorted(rule_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_violation_types = [{"rule": rid, "count": cnt} for rid, cnt in top_types]

        # Issues summary
        issues: List[str] = []

        if files_analyzed == 0:
            issues.append("No Verilog/SystemVerilog files to analyze")
        elif total_violations == 0:
            issues.append("No synthesis safety violations detected - excellent design safety!")
        else:
            issues.append(f"Total synthesis safety violations: {total_violations}")
            issues.append(f"Files with violations: {files_with_violations}")
            issues.append(f"Clean files (no findings): {clean_files}")

            if critical_present:
                issues.append(
                    f"CRITICAL: {severity_breakdown.get('critical', 0)} critical design issue(s) found "
                    f"across {files_with_critical} file(s) - immediate attention required!"
                )
            if severity_breakdown.get("high", 0) > 0:
                issues.append(
                    f"HIGH: {severity_breakdown.get('high', 0)} high-severity issue(s) in "
                    f"{files_with_high} file(s)"
                )
            if severity_breakdown.get("medium", 0) > 0:
                issues.append(
                    f"MEDIUM: {severity_breakdown.get('medium', 0)} medium-severity issue(s) detected"
                )

            for rid, cnt in top_types[:5]:
                desc = next((r["desc"] for r in rules if r["id"] == rid), rid)
                issues.append(f"{rid}: {desc} — {cnt} occurrence(s)")

        metrics = {
            "files_analyzed": files_analyzed,
            "total_violations": total_violations,
            "risk_points": risk_points,
            "critical_rule_present": critical_present,
            "top_violation_types": top_violation_types,
            "violations_by_file": violations_by_file,
            "severity_breakdown": severity_breakdown,
            "rule_counts": dict(rule_counts),
            "files_with_violations": files_with_violations,
            "clean_files": clean_files,
            "files_with_critical": files_with_critical,
            "files_with_high": files_with_high,
        }

        return {
            "score": round(score, 1),
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"
