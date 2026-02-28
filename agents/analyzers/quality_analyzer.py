"""
Verilog/SystemVerilog code quality analysis
"""

import os
import re
from typing import Dict, List, Any, Tuple
from collections import Counter


class QualityAnalyzer:
    """
    Analyzes Verilog/SystemVerilog code quality focusing on:
    - HDL anti-patterns (blocking in sequential, non-blocking in combinational)
    - Incomplete sensitivity lists
    - Latch inference detection
    - Implicit net declarations
    - Delay usage in synthesizable RTL
    - Multiple drivers on same signal
    - Missing default cases
    - Code style consistency (line length, tabs, trailing whitespace)
    """

    # Verilog/SystemVerilog file extensions
    V_EXTS = {".v", ".sv"}
    VH_EXTS = {".vh", ".svh"}

    def __init__(self, codebase_path: str = None, project_root: str = None):
        """Initialize quality analyzer."""
        self.codebase_path = codebase_path or os.getcwd()
        self.project_root = project_root or os.getcwd()
        self._file_cache: List[Dict[str, Any]] = []

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog code quality.

        Args:
            file_cache: List of processed Verilog files with:
                - "suffix": file extension
                - "source": file contents
                - one of: "rel_path", "path", "file_relative_path", "file_name"

        Returns:
            Quality analysis results with score, grade, metrics, issues, and violations.
        """
        self._file_cache = file_cache or []
        return self._calculate_quality_score()

    def _calculate_quality_score(self) -> Dict[str, Any]:
        """
        Calculate Verilog/SystemVerilog code quality score.

        Detects HDL anti-patterns and style issues.
        Each violation includes file path, line, column, rule code, severity, and snippet.
        Score is severity-weighted and normalized by number of files.
        """
        if not self._file_cache:
            return {
                "score": 0.0,
                "grade": "F",
                "issues": ["No files cached"],
                "metrics": {
                    "total_violations": 0,
                    "violations_per_file": 0.0,
                    "files_analyzed": 0,
                    "rule_counts": {},
                    "severity_counts": {},
                },
                "violations": [],
            }

        # Helper: relative path
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

        # Helper: map absolute offset to (line, column)
        def _line_col_from_abs(lines: List[str], abs_pos: int) -> Tuple[int, int]:
            running = 0
            for i, line in enumerate(lines):
                next_running = running + len(line) + 1
                if next_running > abs_pos:
                    col = abs_pos - running + 1
                    return i, max(1, col)
                running = next_running
            return max(0, len(lines) - 1), 1

        # Filter Verilog/SystemVerilog files
        all_exts = {ext.lower() for ext in (self.V_EXTS | self.VH_EXTS)}
        v_files: List[Dict[str, Any]] = []
        for f in self._file_cache:
            suffix = (f.get("suffix") or "").lower()
            if suffix in all_exts:
                v_files.append(f)

        print(
            f"DEBUG quality: Found {len(v_files)} Verilog files out of "
            f"{len(self._file_cache)} total"
        )

        if not v_files:
            return {
                "score": 100.0,
                "grade": "A",
                "metrics": {
                    "total_violations": 0,
                    "violations_per_file": 0.0,
                    "files_analyzed": 0,
                    "rule_counts": {},
                    "severity_counts": {},
                },
                "issues": [
                    "No Verilog/SystemVerilog files found for quality analysis. "
                    f"Extensions checked: {sorted(self.V_EXTS | self.VH_EXTS)}"
                ],
                "violations": [],
            }

        # HDL anti-pattern rules
        hdl_patterns = [
            # Blocking assignments in sequential blocks
            (r"always\s*@\s*\(posedge|negedge\).*?begin.*?[^<]\s*=\s*", "HDL001", "Blocking assignment in sequential always block (use <=)", "high"),
            # Non-blocking in combinational blocks
            (r"always\s*@\s*\*\s*begin.*?<=", "HDL002", "Non-blocking assignment in combinational block (use =)", "high"),
            # Incomplete sensitivity list
            (r"always\s*@\s*\([a-zA-Z0-9_\s,]*\)\s*begin", "HDL003", "Potential incomplete sensitivity list (consider always @*)", "medium"),
            # Initial blocks in synthesizable code
            (r"initial\s+begin", "HDL004", "Initial block may not synthesize (use reset if needed)", "medium"),
            # Implicit net declarations
            (r"(?:input|output|wire|reg)\s+[a-zA-Z0-9_]+\s+([a-zA-Z0-9_]+).*?[a-zA-Z0-9_]+\s*=", "HDL005", "Implicit net declaration without explicit type", "low"),
            # Delays in synthesizable RTL
            (r"#\s*\d+", "HDL006", "Time delay in synthesizable RTL (#delay)", "high"),
            # Multiple drivers (continuous assignments or multiple always blocks on same signal)
            (r"assign\s+[a-zA-Z0-9_]+.*?;.*?assign\s+\1", "HDL007", "Multiple drivers detected on same signal", "critical"),
            # Missing default in case
            (r"case\s*\([^)]+\)(?!.*?default)", "HDL008", "Case statement missing default clause", "medium"),
        ]

        # Style rules
        STYLE_LINE_LEN_RULE = "STYLE-LINELENGTH"
        STYLE_TODO_RULE = "STYLE-TODO"
        STYLE_TABS_RULE = "STYLE-TABS"
        STYLE_TRAILWS_RULE = "STYLE-TRAILWS"

        # Compile regexes
        compiled_hdl = [
            (re.compile(p, re.MULTILINE | re.DOTALL), code, msg, sev)
            for p, code, msg, sev in hdl_patterns
        ]

        violations: List[Dict[str, Any]] = []
        issues: List[str] = []

        # Severity weights for scoring
        severity_weights = {
            "critical": 5.0,
            "high": 4.0,
            "medium": 2.0,
            "low": 1.0,
        }

        # Aggregation counters
        rule_counts: Counter = Counter()
        severity_counts: Counter = Counter()
        severity_weight_sum = 0.0

        print(f"DEBUG quality: Starting analysis of {len(v_files)} files")

        for entry in v_files:
            source = entry.get("source", "") or ""
            if not source.strip():
                continue

            relpath = _rel_path(entry)
            lines = source.splitlines()
            print(
                f"DEBUG quality: Analyzing {relpath} "
                f"({len(lines)} lines, {len(source)} chars)"
            )

            # Style checks
            long_line_count = 0
            tab_count = 0
            trail_ws_count = 0
            todo_count = 0

            for i, line in enumerate(lines, start=1):
                if len(line) > 120:
                    violations.append(
                        {
                            "rule": STYLE_LINE_LEN_RULE,
                            "severity": "low",
                            "message": f"Line exceeds 120 chars ({len(line)})",
                            "file": relpath,
                            "line": i,
                            "column": 1,
                            "snippet": line[:200],
                        }
                    )
                    rule_counts[STYLE_LINE_LEN_RULE] += 1
                    severity_counts["low"] += 1
                    severity_weight_sum += severity_weights["low"]
                    long_line_count += 1

                if "\t" in line:
                    violations.append(
                        {
                            "rule": STYLE_TABS_RULE,
                            "severity": "low",
                            "message": "Tab character used; prefer spaces per coding standard",
                            "file": relpath,
                            "line": i,
                            "column": line.find("\t") + 1,
                            "snippet": line[:200],
                        }
                    )
                    rule_counts[STYLE_TABS_RULE] += 1
                    severity_counts["low"] += 1
                    severity_weight_sum += severity_weights["low"]
                    tab_count += 1

                if line.rstrip() != line:
                    violations.append(
                        {
                            "rule": STYLE_TRAILWS_RULE,
                            "severity": "low",
                            "message": "Trailing whitespace",
                            "file": relpath,
                            "line": i,
                            "column": len(line),
                            "snippet": line[:200],
                        }
                    )
                    rule_counts[STYLE_TRAILWS_RULE] += 1
                    severity_counts["low"] += 1
                    severity_weight_sum += severity_weights["low"]
                    trail_ws_count += 1

                if re.search(r"(?://|/\*).*(TODO|FIXME|HACK)|/\*.*(TODO|FIXME|HACK).*?\*/", line, re.IGNORECASE):
                    violations.append(
                        {
                            "rule": STYLE_TODO_RULE,
                            "severity": "low",
                            "message": "TODO/FIXME/HACK comment",
                            "file": relpath,
                            "line": i,
                            "column": 1,
                            "snippet": line.strip()[:200],
                        }
                    )
                    rule_counts[STYLE_TODO_RULE] += 1
                    severity_counts["low"] += 1
                    severity_weight_sum += severity_weights["low"]
                    todo_count += 1

            # HDL anti-patterns
            hdl_count = 0
            for cre, code, msg, sev in compiled_hdl:
                for m in cre.finditer(source):
                    li, col = _line_col_from_abs(lines, m.start())
                    violations.append(
                        {
                            "rule": code,
                            "severity": sev,
                            "message": msg,
                            "file": relpath,
                            "line": li + 1,
                            "column": col,
                            "snippet": lines[li].strip()[:200] if li < len(lines) else "",
                        }
                    )
                    rule_counts[code] += 1
                    severity_counts[sev] += 1
                    severity_weight_sum += severity_weights.get(sev, 1.0)
                    hdl_count += 1

            print(
                f"DEBUG quality: File {relpath} - Long lines: {long_line_count}, "
                f"TABS: {tab_count}, TrailingWS: {trail_ws_count}, TODOs: {todo_count}, "
                f"HDL anti-patterns: {hdl_count}"
            )

        total_violations = sum(rule_counts.values())
        analyzed_files = len(v_files)
        violations_per_file = total_violations / max(1, analyzed_files)

        print(
            f"DEBUG quality: Total violations: {total_violations}, "
            f"severity-weight sum: {severity_weight_sum:.1f}"
        )

        # Scoring
        avg_severity_weight_per_file = severity_weight_sum / max(1, analyzed_files)
        score = 100.0 - 2.0 * avg_severity_weight_per_file
        score = max(0.0, min(100.0, score))

        if violations_per_file > 10:
            issues.append(
                f"High violation density: {violations_per_file:.1f} issues per Verilog file"
            )
        if severity_counts.get("critical", 0) > 0:
            issues.append(
                f"{severity_counts['critical']} critical issue(s) detected (multiple drivers, delays in RTL)"
            )
        if severity_counts.get("high", 0) > 0:
            issues.append(
                f"{severity_counts['high']} high severity issues detected (blocking/non-blocking misuse)"
            )
        if total_violations == 0 and analyzed_files > 0:
            issues.append("No quality violations detected - excellent code quality!")

        grade = self._score_to_grade(score)

        metrics = {
            "total_violations": total_violations,
            "violations_per_file": round(violations_per_file, 2),
            "files_analyzed": analyzed_files,
            "rule_counts": dict(rule_counts),
            "severity_counts": dict(severity_counts),
            "avg_severity_weight_per_file": round(avg_severity_weight_per_file, 2),
        }

        return {
            "score": round(score, 1),
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
            "violations": violations,
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
