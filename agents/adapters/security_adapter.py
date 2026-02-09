"""Security vulnerability scanning using Flawfinder."""

import csv
import logging
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from agents.adapters.base_adapter import BaseStaticAdapter

# Try to import flawfinder at module level
try:
    import flawfinder
    FLAWFINDER_AVAILABLE = True
except ImportError:
    FLAWFINDER_AVAILABLE = False


class SecurityAdapter(BaseStaticAdapter):
    """
    Scans C/C++ code for security vulnerabilities using Flawfinder.

    Analyzes source code for common security issues, CWE violations,
    and dangerous function calls. Reports findings by severity and CWE.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize security adapter.

        Args:
            debug: Enable debug logging if True.
        """
        super().__init__("security", debug=debug)
        self.flawfinder_available = FLAWFINDER_AVAILABLE
        if not self.flawfinder_available:
            self.logger.warning(
                "Flawfinder not available. Install with: pip install flawfinder"
            )

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        ccls_navigator: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze C/C++ files for security vulnerabilities.

        Args:
            file_cache: List of file entries with "file_relative_path" and "source" keys.
            ccls_navigator: Optional CCLS navigator (unused here).
            dependency_graph: Optional dependency graph (unused here).

        Returns:
            Standard analysis result dict with score, grade, metrics, issues, details.
        """
        # Step 1: Check tool availability
        if not self.flawfinder_available:
            return self._handle_tool_unavailable(
                "Flawfinder", "Install with: pip install flawfinder"
            )

        # Step 2: Filter to C/C++ files
        c_cpp_suffixes = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx")
        cpp_files = [
            entry
            for entry in file_cache
            if entry.get("file_relative_path", "").endswith(c_cpp_suffixes)
        ]

        if not cpp_files:
            return self._empty_result("No C/C++ files to analyze")

        # Step 3: Scan each file with flawfinder
        all_findings = []
        for entry in cpp_files:
            file_path = entry.get("file_relative_path", "unknown")
            source_code = entry.get("source", "")

            try:
                # Write source to temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".c", delete=False
                ) as tmp:
                    tmp.write(source_code)
                    tmp_path = tmp.name

                # Run flawfinder on the temp file
                result = subprocess.run(
                    ["flawfinder", "--csv", "--columns", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Parse CSV output (handle exit code 1 as success when findings exist)
                findings = self._parse_flawfinder_csv(
                    result.stdout, file_path, source_code
                )
                all_findings.extend(findings)

                if self.debug:
                    self.logger.debug(
                        f"Scanned {file_path}: found {len(findings)} issues"
                    )

            except subprocess.TimeoutExpired:
                self.logger.error(f"Flawfinder timeout on {file_path}")
            except Exception as e:
                self.logger.error(f"Error scanning {file_path}: {e}")
            finally:
                # Clean up temp file
                try:
                    import os
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if not all_findings:
            return self._empty_result("No security issues found")

        # Step 4: Categorize findings by severity
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        cwe_breakdown = {}
        details = []

        for finding in all_findings:
            level = finding["level"]

            if level == 5:
                critical_count += 1
                severity = "critical"
            elif level == 4:
                high_count += 1
                severity = "high"
            elif level == 3:
                medium_count += 1
                severity = "medium"
            else:  # 1-2
                low_count += 1
                severity = "low"

            # Track CWE breakdown
            cwe = finding.get("cwe", "")
            if cwe:
                cwe_breakdown[cwe] = cwe_breakdown.get(cwe, 0) + 1

            # Create detail entry
            description = finding["warning"]
            category = finding.get("category", "security")

            detail = self._make_detail(
                file=finding["file"],
                function=finding.get("context", ""),
                line=finding["line"],
                description=description,
                severity=severity,
                category=category,
                cwe=cwe,
            )
            details.append(detail)

        # Step 5: Calculate score
        total_findings = len(all_findings)
        score = 100.0
        score -= critical_count * 10
        score -= high_count * 5
        score -= medium_count * 2
        score -= low_count * 1
        score = max(0, min(100, score))

        # Step 6: Compute metrics
        top_categories = self._get_top_categories(details, top_n=5)

        metrics = {
            "tool_available": True,
            "files_analyzed": len(cpp_files),
            "total_findings": total_findings,
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "cwe_breakdown": cwe_breakdown,
            "top_categories": top_categories,
        }

        # Step 7: Build issues list
        issues = []
        if critical_count > 0:
            issues.append(f"Found {critical_count} critical security issue(s)")
        if high_count > 0:
            issues.append(f"Found {high_count} high-severity security issue(s)")
        if medium_count > 0:
            issues.append(f"Found {medium_count} medium-severity security issue(s)")
        if low_count > 0:
            issues.append(f"Found {low_count} low-severity security issue(s)")
        if not issues:
            issues = ["No security issues detected"]

        grade = self._score_to_grade(score)

        return {
            "score": score,
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _parse_flawfinder_csv(
        self, csv_output: str, original_file_path: str, source_code: str
    ) -> List[Dict[str, Any]]:
        """
        Parse Flawfinder CSV output.

        Expected columns: File, Line, Column, Level, Category, Name, Warning, Suggestion, Note, CWEs, Context, Fingerprint

        Args:
            csv_output: Raw CSV text from flawfinder
            original_file_path: Original file path to use (flawfinder may have temp path)
            source_code: Source code for extracting context

        Returns:
            List of finding dicts with file, line, level, warning, cwe, category, context.
        """
        findings = []
        try:
            reader = csv.DictReader(csv_output.strip().split("\n"))
            if reader is None:
                return findings

            for row in reader:
                if not row or not row.get("File"):
                    continue

                try:
                    line_num = int(row.get("Line", 0))
                    level = int(row.get("Level", 1))
                except ValueError:
                    continue

                # Extract CWE from CWEs column (may contain multiple, e.g., "CWE-78, CWE-94")
                cwe_str = row.get("CWEs", "")
                primary_cwe = ""
                if cwe_str:
                    # Take the first CWE
                    parts = cwe_str.split(",")
                    if parts:
                        primary_cwe = parts[0].strip()

                finding = {
                    "file": original_file_path,
                    "line": line_num,
                    "level": level,
                    "warning": row.get("Warning", "Unknown warning"),
                    "cwe": primary_cwe,
                    "category": row.get("Category", "security"),
                    "context": row.get("Context", ""),
                    "suggestion": row.get("Suggestion", ""),
                    "name": row.get("Name", ""),
                }
                findings.append(finding)

        except Exception as e:
            self.logger.error(f"Error parsing flawfinder CSV: {e}")

        return findings

    @staticmethod
    def _get_top_categories(
        details: List[Dict[str, Any]], top_n: int = 5
    ) -> Dict[str, int]:
        """
        Get top security issue categories.

        Args:
            details: List of detail dicts from analysis
            top_n: Number of top categories to return

        Returns:
            Dict mapping category name to count
        """
        category_counts = {}
        for detail in details:
            cat = detail.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count descending and take top N
        sorted_cats = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_cats[:top_n])
