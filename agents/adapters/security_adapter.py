"""HDL lint and design rule violation scanning using Verilator/Verible (with regex fallback)."""

import csv
import logging
import re
import subprocess
import tempfile
import shutil
import os
from typing import Any, Dict, List, Optional, Tuple

from agents.adapters.base_adapter import BaseStaticAdapter

import shutil
VERILATOR_PATH = shutil.which("verilator")
VERILATOR_AVAILABLE = VERILATOR_PATH is not None
VERIBLE_PATH = shutil.which("verible-verilog-lint")
VERIBLE_AVAILABLE = VERIBLE_PATH is not None


# ── Regex-based HDL anti-pattern database ───────────────────────────────────
# Format: (pattern, level 0-5, DRC, category, warning message)
_HDL_VIOLATIONS: List[Tuple[re.Pattern, int, str, str, str]] = [
    # HDL-001: Blocking assignment in sequential logic (always @(posedge/negedge))
    (re.compile(r'always\s*@\s*\((?:posedge|negedge).*?\).*?[^<]='), 5, "HDL-001", "sequential",
     "Blocking assignment in sequential logic; use non-blocking (<=) instead."),

    # HDL-002: Non-blocking assignment in combinational logic (always @(*) or always_comb)
    (re.compile(r'(?:always\s*@\s*\(\*\)|always_comb).*?<='), 4, "HDL-002", "combinational",
     "Non-blocking assignment in combinational logic; use blocking (=) instead."),

    # HDL-003: Incomplete if/case without else/default (latch inference)
    (re.compile(r'always\s*@.*?\n\s*if\s*\(.*?\)\s*.*?(?!else\s)'), 4, "HDL-003", "latch_inference",
     "Incomplete if statement without else; may infer unintended latches."),

    # HDL-005: Initial block in synthesizable code
    (re.compile(r'\binitial\s*begin'), 3, "HDL-005", "synthesis",
     "Initial block may not be synthesizable; verify intention."),

    # HDL-006: Implicit net declarations
    (re.compile(r'^\s*(?!.*\b(?:wire|reg|input|output|logic|bit|integer|real|inout)\b)\w+\s+\w+\s*[,;]', re.MULTILINE), 2, "HDL-006", "declaration",
     "Implicit net declaration detected; explicitly declare all signals."),

    # HDL-007: Delay in RTL code (not synthesizable)
    (re.compile(r'#\s*\d+'), 3, "HDL-007", "delay",
     "Time delay (#) in RTL code is not synthesizable."),

    # HDL-008: Case statement without default
    (re.compile(r'case\s*\(.*?\).*?(?!default)'), 3, "HDL-008", "completeness",
     "Case statement missing default clause; may infer latches."),

    # HDL-009: Missing generate block label
    (re.compile(r'generate\s*\n\s*for|if\s*\('), 2, "HDL-009", "generate",
     "Generate block should have explicit label for clarity."),

    # HDL-011: Asynchronous reset without synchronizer
    (re.compile(r'always\s*@\s*\(.*?negedge\s+rst'), 3, "HDL-011", "reset",
     "Asynchronous reset detected; ensure proper de-assertion synchronization."),

    # HDL-012: Single-bit CDC (Clock Domain Crossing) without synchronizer
    (re.compile(r'assign\s+\w+\s*=\s*\w+_clk'), 4, "HDL-012", "cdc",
     "Signal crossing clock domain detected; add synchronizer for metastability."),

    # HDL-013: X assignment in RTL
    (re.compile(r"=\s*\d+'[xX]"), 3, "HDL-013", "simulation",
     "X (unknown) value assignment detected; may hide bugs in synthesis."),

    # HDL-014: Floating input port
    (re.compile(r'input\s+.*?;(?!\s*//\s*(?:pull|tie|unused))'), 2, "HDL-014", "port",
     "Input port may be floating; ensure proper termination or tie-off."),

    # HDL-015: Width mismatch in assignment
    (re.compile(r'=\s*\d+\'(?:h|b|d)\d+'), 3, "HDL-015", "width",
     "Width mismatch detected in assignment; verify bit widths match."),

    # HDL-016: Signed/unsigned mixing
    (re.compile(r'signed.*?unsigned|unsigned.*?signed'), 2, "HDL-016", "type",
     "Signed and unsigned types mixed; may cause unexpected behavior."),

    # HDL-018: Multiple always blocks driving same signal
    (re.compile(r'always.*?=\s*\w+'), 3, "HDL-018", "fan_in",
     "Signal driven by multiple always blocks; potential race condition."),

    # HDL-019: Clock used as data
    (re.compile(r'data.*?clk|clk.*?data'), 4, "HDL-019", "clock_usage",
     "Clock signal used as data; may cause timing violations."),

    # HDL-020: Data used as clock
    (re.compile(r'\.clk\s*\([^c]'), 4, "HDL-020", "clock_usage",
     "Data signal used as clock; verify this is intentional."),
]


class LintAdapter(BaseStaticAdapter):
    """
    Scans Verilog/SystemVerilog code for lint warnings and design rule violations.

    Analyzes source code for common HDL issues, DRC violations, and anti-patterns.
    Reports findings by severity and DRC code.

    Falls back to regex-based scanning when Verilator/Verible are not installed.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize lint adapter.

        Args:
            debug: Enable debug logging if True.
        """
        super().__init__("lint", debug=debug)

        self.verilator_available = VERILATOR_AVAILABLE
        self.verible_available = VERIBLE_AVAILABLE

        if not (self.verilator_available or self.verible_available):
            self.logger.warning(
                "Verilator and Verible not found — using regex fallback. "
                "For best results: install Verilator or Verible"
            )

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        verible_parser: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog files for lint warnings and DRC violations.

        Args:
            file_cache: List of file entries with "file_relative_path" and "source" keys.
            verible_parser: Optional Verible parser (unused here).
            dependency_graph: Optional dependency graph (unused here).

        Returns:
            Standard analysis result dict with score, grade, metrics, issues, details.
        """
        using_fallback = not (self.verilator_available or self.verible_available)

        # Filter to Verilog/SystemVerilog files
        verilog_suffixes = (".v", ".sv", ".svh", ".vh")
        verilog_files = [
            entry
            for entry in file_cache
            if entry.get("file_relative_path", "").lower().endswith(verilog_suffixes)
        ]

        if not verilog_files:
            return self._empty_result("No Verilog/SystemVerilog files to analyze")

        # Scan each file
        all_findings = []
        files_scanned = 0

        for entry in verilog_files:
            file_path = entry.get("file_relative_path", "unknown")
            source_code = entry.get("source", "")

            if not source_code.strip():
                continue

            if self.verilator_available or self.verible_available:
                # Try Verilator or Verible
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".sv", delete=False, encoding='utf-8'
                    ) as tmp:
                        tmp.write(source_code)
                        tmp_path = tmp.name

                    findings = None

                    # Try Verilator first
                    if self.verilator_available:
                        findings = self._run_verilator(tmp_path, file_path, source_code)

                    # Fall back to Verible if Verilator didn't produce results
                    if not findings and self.verible_available:
                        findings = self._run_verible(tmp_path, file_path, source_code)

                    if findings:
                        all_findings.extend(findings)

                    files_scanned += 1

                except subprocess.TimeoutExpired:
                    self.logger.error(f"Tool timeout on {file_path}")
                except Exception as e:
                    self.logger.error(f"Error scanning {file_path}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            else:
                # Regex fallback path
                try:
                    findings = self._regex_scan_file(file_path, source_code)
                    if findings:
                        all_findings.extend(findings)
                    files_scanned += 1
                except Exception as e:
                    self.logger.error(f"Error regex-scanning {file_path}: {e}")

        # Step 4: Handle "No Issues Found" case explicitly (FIX for 0 Score)
        if not all_findings:
            if self.debug:
                self.logger.info("Flawfinder scan completed: 0 issues found.")
            
            return {
                "score": 100.0,
                "grade": "A",
                "metrics": {
                    "tool_available": True,
                    "files_analyzed": files_scanned,
                    "total_findings": 0,
                    "critical_count": 0,
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": 0,
                    "drc_breakdown": {},
                    "top_categories": {}
                },
                "issues": ["No lint issues detected"],
                "details": [],
                "tool_available": True,
            }

        # Categorize findings by severity
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        drc_breakdown = {}
        details = []

        for finding in all_findings:
            level = finding["level"]

            if level >= 5:
                critical_count += 1
                severity = "critical"
            elif level == 4:
                high_count += 1
                severity = "high"
            elif level == 3:
                medium_count += 1
                severity = "medium"
            else:  # 0-2
                low_count += 1
                severity = "low"

            # Track DRC breakdown
            drc = finding.get("drc", "")
            if drc:
                drc_breakdown[drc] = drc_breakdown.get(drc, 0) + 1

            # Create detail entry
            description = finding["warning"]
            category = finding.get("category", "lint")

            detail = self._make_detail(
                file=finding["file"],
                module=finding.get("context", ""),
                line=finding["line"],
                description=description,
                severity=severity,
                category=category,
                drc=drc,
            )
            details.append(detail)

        # Calculate score
        score = 100.0
        score -= critical_count * 15
        score -= high_count * 5
        score -= medium_count * 2
        score -= low_count * 0.5
        score = max(0.0, min(100.0, score))

        # Compute metrics
        top_categories = self._get_top_categories(details, top_n=5)

        metrics = {
            "tool_available": True,
            "files_analyzed": files_scanned,
            "total_findings": len(all_findings),
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "drc_breakdown": drc_breakdown,
            "top_categories": top_categories,
        }

        # Build issues list
        issues = []
        if critical_count > 0:
            issues.append(f"Found {critical_count} critical lint issue(s)")
        if high_count > 0:
            issues.append(f"Found {high_count} high-severity lint issue(s)")
        if medium_count > 0:
            issues.append(f"Found {medium_count} medium-severity lint issue(s)")
        if low_count > 0:
            issues.append(f"Found {low_count} low-severity lint issue(s)")

        if not issues:
            issues = ["No significant lint issues detected"]

        grade = self._score_to_grade(score)

        return {
            "score": score,
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _run_verilator(
        self, tmp_path: str, file_path: str, source_code: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Run Verilator lint on file."""
        try:
            cmd = [VERILATOR_PATH, '--lint-only', '--Wall', '-Wno-fatal', tmp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0 and not result.stderr:
                return None

            # Parse Verilator warning output
            findings = self._parse_verilator_output(result.stderr, file_path)
            return findings if findings else None

        except Exception as e:
            if self.debug:
                self.logger.debug(f"Verilator analysis failed: {e}")
            return None

    def _run_verible(
        self, tmp_path: str, file_path: str, source_code: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Run Verible lint on file."""
        try:
            cmd = [VERIBLE_PATH, tmp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Verible returns findings in stdout
            findings = self._parse_verible_output(result.stdout, file_path)
            return findings if findings else None

        except Exception as e:
            if self.debug:
                self.logger.debug(f"Verible analysis failed: {e}")
            return None

    def _parse_verilator_output(
        self, output: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse Verilator warning output."""
        findings = []
        for line in output.splitlines():
            # Verilator format: filename:linenum:colnum: <level>: <message>
            match = re.match(r'^.*?:(\d+):\d+:\s*(\w+):\s*(.*?)$', line)
            if match:
                line_num = int(match.group(1))
                level_str = match.group(2).lower()
                message = match.group(3)

                level = 3 if level_str == 'warning' else (5 if level_str == 'error' else 2)

                finding = {
                    "file": file_path,
                    "line": line_num,
                    "level": level,
                    "warning": message,
                    "drc": "HDL-LINT",
                    "category": "lint",
                    "context": "",
                }
                findings.append(finding)

        return findings

    def _parse_verible_output(
        self, output: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse Verible lint output."""
        findings = []
        for line in output.splitlines():
            # Verible format: filename:linenum:colnum: [level] message
            match = re.match(r'^.*?:(\d+):\d+:\s*\[(\w+)\]\s*(.*?)$', line)
            if match:
                line_num = int(match.group(1))
                level_str = match.group(2).lower()
                message = match.group(3)

                level = 3 if level_str == 'warning' else (5 if level_str == 'error' else 2)

                finding = {
                    "file": file_path,
                    "line": line_num,
                    "level": level,
                    "warning": message,
                    "drc": "HDL-LINT",
                    "category": "lint",
                    "context": "",
                }
                findings.append(finding)

        return findings

    def _regex_scan_file(
        self, file_path: str, source_code: str
    ) -> List[Dict[str, Any]]:
        """Scan a single file for HDL anti-patterns using regex patterns."""
        findings = []
        lines = source_code.splitlines()
        for line_idx, line in enumerate(lines, start=1):
            # Skip comments (simple heuristic)
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            for pattern, level, drc, category, warning in _HDL_VIOLATIONS:
                if pattern.search(line):
                    findings.append({
                        "file": file_path,
                        "line": line_idx,
                        "level": level,
                        "warning": warning,
                        "drc": drc,
                        "category": category,
                        "context": stripped[:120],
                    })
        return findings

    @staticmethod
    def _get_top_categories(
        details: List[Dict[str, Any]], top_n: int = 5
    ) -> Dict[str, int]:
        """Get top security issue categories."""
        category_counts = {}
        for detail in details:
            cat = detail.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        sorted_cats = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_cats[:top_n])