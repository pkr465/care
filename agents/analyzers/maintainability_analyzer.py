"""
Verilog/SystemVerilog maintainability analysis
"""

import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MaintainabilityAnalyzer:
    """
    MaintainabilityAnalyzer: Verilog/SystemVerilog maintainability analysis.

    - Per-file analysis:
      - LOC (total / code / comment / blank)
      - Comment ratio
      - Cyclomatic complexity (heuristic)
      - Halstead volume (approximate)
      - Maintainability Index (SEI-style, scaled to 0–100)
      - Module and port documentation
      - Include guard checks (for headers)
      - HDL anti-patterns
      - Formatting issues: long lines, tabs, trailing whitespace, TODO/FIXME
      - Missing default in case statements

    - Aggregated metrics and score calculation
    """

    # Patterns
    _HDL_ANTIPATTERN = [
        re.compile(r"always\s*@\s*\(posedge.*?[^<]\s*=\s*"),  # Blocking in sequential
        re.compile(r"always\s*@\s*\*.*?<="),  # Non-blocking in combinational
    ]

    _TODO_PATTERN = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)
    _LONG_LINE_LIMIT = 120
    _TAB_PATTERN = re.compile(r"\t")
    _TRAILING_WS_PATTERN = re.compile(r"[ \t]+$")

    def __init__(
        self,
        codebase_path: str,
        project_root: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.codebase_path = Path(codebase_path)
        self.project_root = Path(project_root) if project_root else self.codebase_path
        self.debug = debug

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform maintainability analysis on a file cache.

        Args:
            file_cache: entries with "suffix", "source", "file_relative_path"
            dependency_graph: optional (unused for now)

        Returns:
            {
              "score": float,
              "grade": str,
              "metrics": {...},
              "issues": [str, ...]
            }
        """
        per_file_metrics: List[Dict[str, Any]] = []

        for entry in file_cache:
            suffix = entry.get("suffix", "").lower()
            if suffix not in {".v", ".sv", ".vh", ".svh"}:
                continue

            rel_path = entry.get("file_relative_path") or entry.get("rel_path") or entry.get("path")
            source = entry.get("source", "")

            try:
                file_metrics = self._analyze_single_file(rel_path, suffix, source)
                per_file_metrics.append(file_metrics)
            except Exception as e:
                if self.debug:
                    print(f"[MaintainabilityAnalyzer] Error analyzing {rel_path}: {e}")

        aggregated, issues, score = self._aggregate_metrics(per_file_metrics)
        grade = self._score_to_grade(score)

        return {
            "score": round(score, 1),
            "grade": grade,
            "metrics": aggregated,
            "issues": issues,
        }

    def _analyze_single_file(self, rel_path: str, suffix: str, source: str) -> Dict[str, Any]:
        lines = source.splitlines()
        total_lines = len(lines)

        # Line classification
        loc_total, loc_code, loc_comment, loc_blank = self._classify_lines(lines)
        comment_ratio = loc_comment / max(1, total_lines)

        # Strip comments and strings
        code_no_comments, code_no_comments_or_strings = self._strip_comments_and_strings(source)

        # Cyclomatic complexity
        cyclomatic = self._compute_cyclomatic_complexity(code_no_comments_or_strings)

        # Halstead volume
        halstead_volume = self._compute_halstead_volume(code_no_comments_or_strings)

        # Maintainability Index
        mi = self._compute_maintainability_index(
            volume=halstead_volume,
            complexity=cyclomatic,
            loc=loc_code if loc_code > 0 else total_lines,
        )

        # Module documentation
        module_documented = self._check_module_documented(source)

        # Include guards (for headers)
        has_include_guard = False
        include_guard_issue = ""
        if suffix in {".vh", ".svh"}:
            has_include_guard = self._check_include_guard(lines)
            if not has_include_guard:
                include_guard_issue = "Missing `ifndef/`define/`endif include guard"

        # HDL anti-patterns
        hdl_hits = self._scan_hdl_antipatterns(code_no_comments_or_strings)

        # Formatting
        formatting_issues = self._scan_formatting(lines)

        # Case default checks
        case_missing_default = self._check_case_default(code_no_comments_or_strings)

        file_metrics: Dict[str, Any] = {
            "file": rel_path,
            "suffix": suffix,
            "total_lines": total_lines,
            "code_lines": loc_code,
            "comment_lines": loc_comment,
            "blank_lines": loc_blank,
            "comment_ratio": comment_ratio,
            "cyclomatic_complexity": cyclomatic,
            "halstead_volume": halstead_volume,
            "maintainability_index": mi,
            "module_documented": module_documented,
            "has_include_guard": has_include_guard,
            "include_guard_issue": include_guard_issue,
            "hdl_antipattern_hits": hdl_hits,
            "formatting_issues": formatting_issues,
            "case_missing_default": case_missing_default,
        }

        return file_metrics

    def _classify_lines(self, lines: List[str]) -> Tuple[int, int, int, int]:
        total = len(lines)
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        in_block_comment = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
                continue

            if in_block_comment:
                comment_lines += 1
                if "*/" in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith("/*"):
                comment_lines += 1
                if "*/" not in stripped:
                    in_block_comment = True
                continue

            if stripped.startswith("//"):
                comment_lines += 1
                continue

            code_lines += 1

        return total, code_lines, comment_lines, blank_lines

    def _strip_comments_and_strings(self, source: str) -> Tuple[str, str]:
        """Remove comments and strings, preserving newlines."""
        def _block_replacer(match: re.Match) -> str:
            text = match.group(0)
            newline_count = text.count("\n")
            return "\n" * newline_count

        # Remove block comments
        no_block = re.sub(r"/\*[\s\S]*?\*/", _block_replacer, source)
        # Remove line comments
        no_line = re.sub(r"//.*?$", "", no_block, flags=re.MULTILINE)

        code_no_comments = no_line

        # Remove strings
        def _string_replacer(match: re.Match) -> str:
            s = match.group(0)
            return '"' + " " * (len(s) - 2) + '"' if len(s) >= 2 else '""'

        no_strings = re.sub(r'"([^"\\]|\\.)*"', _string_replacer, code_no_comments)
        code_no_comments_or_strings = no_strings

        return code_no_comments, code_no_comments_or_strings

    def _compute_cyclomatic_complexity(self, code: str) -> int:
        """Heuristic cyclomatic complexity for Verilog."""
        code = re.sub(r"\belse\s+if\b", "if", code)

        complexity = 1
        decision_keywords = [
            r"\bif\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bcase\b",
            r"\bforever\b",
            r"\brepeat\b",
        ]

        for kw in decision_keywords:
            complexity += len(re.findall(kw, code))

        complexity += len(re.findall(r"\?", code))
        complexity += len(re.findall(r"&&|\|\|", code))

        return max(1, complexity)

    def _compute_halstead_volume(self, code: str) -> float:
        """Approximate Halstead Volume."""
        operator_tokens = {
            "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=",
            "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", "=", "+=",
            "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "?",
            ":", "->", ".", ",", ";", "<=" , ":="
        }

        tokens = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|==|!=|<=|>=|&&|\|\||<<=|>>=|<<|>>|->|:=|<=|[-+*/%=!<>&^|?:;,.]",
            code
        )

        operators = [t for t in tokens if t in operator_tokens]
        operands = [t for t in tokens if t not in operator_tokens]

        n1 = len(set(operators))
        n2 = len(set(operands))
        N1 = len(operators)
        N2 = len(operands)

        n = n1 + n2
        N = N1 + N2

        if n == 0 or N == 0:
            return 0.0

        volume = N * self._safe_log2(n)
        return float(volume)

    def _compute_maintainability_index(self, volume: float, complexity: int, loc: int) -> float:
        """SEI-style Maintainability Index."""
        V = max(1.0, volume)
        G = max(1, complexity)
        L = max(1, loc)

        mi_raw = 171.0 - 5.2 * self._safe_ln(V) - 0.23 * G - 16.2 * self._safe_ln(L)
        mi_scaled = mi_raw * (100.0 / 171.0)
        mi_scaled = max(0.0, min(100.0, mi_scaled))

        return mi_scaled

    def _check_module_documented(self, source: str) -> bool:
        """Check if module has documentation."""
        return bool(
            re.search(r"/\*.*?module", source, re.DOTALL | re.IGNORECASE)
            or re.search(r"//.*?module", source, re.IGNORECASE)
        )

    def _check_include_guard(self, lines: List[str]) -> bool:
        """Check for `ifndef/`define/`endif include guard pattern."""
        text = "\n".join(lines[:30])
        return bool(
            re.search(r"`ifndef\s+\w+", text)
            and re.search(r"`define\s+\w+", text)
        )

    def _scan_hdl_antipatterns(self, code: str) -> List[str]:
        hits = []
        for pat in self._HDL_ANTIPATTERN:
            for _m in pat.finditer(code):
                hits.append(pat.pattern)
        return hits

    def _scan_formatting(self, lines: List[str]) -> Dict[str, Any]:
        long_lines = []
        tabs = []
        trailing_ws = []
        todos = []

        for idx, line in enumerate(lines, start=1):
            if len(line) > self._LONG_LINE_LIMIT:
                long_lines.append(idx)
            if self._TAB_PATTERN.search(line):
                tabs.append(idx)
            if self._TRAILING_WS_PATTERN.search(line):
                trailing_ws.append(idx)
            if self._TODO_PATTERN.search(line):
                todos.append(idx)

        return {
            "long_lines": long_lines,
            "tabs": tabs,
            "trailing_whitespace": trailing_ws,
            "todo_lines": todos,
        }

    def _check_case_default(self, code: str) -> bool:
        """Check if case statements are missing defaults."""
        case_matches = re.finditer(r"\bcase\b", code)
        for m in case_matches:
            window = code[m.start() : m.start() + 2000]
            if "default" not in window or window.find("}") < window.find("default"):
                return True
        return False

    @staticmethod
    def _safe_log2(x: float) -> float:
        return math.log2(x) if x > 0 else 0.0

    @staticmethod
    def _safe_ln(x: float) -> float:
        return math.log(x) if x > 0 else 0.0

    def _aggregate_metrics(
        self, per_file: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str], float]:
        issues: List[str] = []

        if not per_file:
            return (
                {
                    "files": [],
                    "avg_mi": 0.0,
                    "min_mi": 0.0,
                    "max_mi": 0.0,
                    "avg_complexity": 0.0,
                    "avg_comment_ratio": 0.0,
                    "total_hdl_antipatterns": 0,
                },
                ["No Verilog files analyzed."],
                0.0,
            )

        mi_values = [f["maintainability_index"] for f in per_file]
        cyclomatic_values = [f["cyclomatic_complexity"] for f in per_file]
        comment_ratios = [f["comment_ratio"] for f in per_file]

        avg_mi = sum(mi_values) / len(mi_values)
        min_mi = min(mi_values)
        max_mi = max(mi_values)
        avg_complexity = sum(cyclomatic_values) / len(cyclomatic_values)
        avg_comment_ratio = sum(comment_ratios) / len(comment_ratios)

        total_hdl_issues = sum(len(f["hdl_antipattern_hits"]) for f in per_file)

        # Issues
        if avg_mi < 50:
            issues.append(
                f"Average maintainability index is low ({avg_mi:.1f}). Refactoring may be needed."
            )
        if total_hdl_issues > 0:
            issues.append(f"Detected {total_hdl_issues} HDL anti-pattern(s).")

        case_issues = [f for f in per_file if f["case_missing_default"]]
        if case_issues:
            issues.append(f"{len(case_issues)} file(s) with case statements missing default.")

        metrics: Dict[str, Any] = {
            "files": per_file,
            "avg_mi": avg_mi,
            "min_mi": min_mi,
            "max_mi": max_mi,
            "avg_complexity": avg_complexity,
            "avg_comment_ratio": avg_comment_ratio,
            "total_hdl_antipatterns": total_hdl_issues,
        }

        score = avg_mi
        if total_hdl_issues > 0:
            score -= min(20.0, 2.0 * total_hdl_issues)
        if avg_complexity > 10:
            score -= min(15.0, (avg_complexity - 10) * 1.0)

        score = max(0.0, min(100.0, score))
        return metrics, issues, score

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"
