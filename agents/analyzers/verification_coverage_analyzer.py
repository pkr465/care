"""
Verilog/SystemVerilog verification coverage analysis
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import Counter


class VerificationCoverageAnalyzer:
    """
    Analyzes Verilog/SystemVerilog verification coverage and testing practices:
    - Test/verification file detection
    - Verification framework identification (UVM, SVA, Cocotb, VUnit)
    - Assertion coverage
    - Coverage artifacts (covergroup detection)
    - Test directory patterns
    """

    V_EXTS = {".v", ".sv"}
    VH_EXTS = {".vh", ".svh"}

    def __init__(self, codebase_path: str = None, project_root: str = None):
        """Initialize verification coverage analyzer."""
        self.codebase_path = codebase_path or os.getcwd()
        self.project_root = project_root or os.getcwd()
        self._file_cache: List[Dict[str, Any]] = []

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog verification coverage and testing practices.

        Args:
            file_cache: List of processed file entries

        Returns:
            Verification coverage analysis results
        """
        self._file_cache = file_cache or []
        return self._calculate_verification_coverage_score()

    def _calculate_verification_coverage_score(self) -> Dict[str, Any]:
        """
        Verilog/SystemVerilog verification coverage heuristic scoring.
        """
        if not getattr(self, "_file_cache", None):
            return {
                "score": 0.0,
                "grade": "F",
                "issues": ["No files cached"],
                "metrics": {},
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

        all_files = self._file_cache

        # Filter Verilog files
        all_exts = {ext.lower() for ext in (self.V_EXTS | self.VH_EXTS)}
        v_files: List[Dict[str, Any]] = []
        for f in self._file_cache:
            suffix = (f.get("suffix") or "").lower()
            if suffix in all_exts:
                v_files.append(f)

        print(
            f"DEBUG verification_coverage: Found {len(v_files)} Verilog files out of "
            f"{len(self._file_cache)} total"
        )

        # Score weights
        W = {
            "testbench_ratio": 0.25,
            "frameworks": 0.20,
            "assertions": 0.25,
            "coverage": 0.20,
            "tools": 0.10,
        }

        def _is_rtl_file(entry: Dict[str, Any]) -> bool:
            name = (entry.get("file_name") or "").lower()
            path = _rel_path(entry).lower()
            return not any(p in path or p in name for p in ["_tb", "test_", "tb_", "/tb/", "bench"])

        def _likely_testbench_file(entry: Dict[str, Any]) -> bool:
            """Detect testbench/verification files."""
            name = (entry.get("file_name") or "").lower()
            path = _rel_path(entry).lower()
            patterns = ["_tb", "test_", "tb_", ".test", "_test"]
            paths = ["/tb/", "/testbench/", "/test/", "/verification/", "/sim/"]
            return any(p in name for p in patterns) or any(p in path for p in paths)

        # Framework detection patterns
        FW_PATTERNS = {
            "uvm": [
                r"uvm_test\b",
                r"uvm_env\b",
                r"uvm_agent\b",
                r"uvm_driver\b",
                r"import\s+uvm_pkg",
            ],
            "sva": [
                r"assert\s+property\b",
                r"assume\s+property\b",
                r"cover\s+property\b",
                r"\$assert\b",
            ],
            "cocotb": [
                r"import\s+cocotb",
                r"@cocotb\.test",
            ],
            "vunit": [
                r"run_all_in_parallel",
                r"tb_entity_pkg",
            ],
        }

        ASSERTION_PATTERNS = [
            r"assert\s+property\b",
            r"assume\s+property\b",
            r"cover\s+property\b",
            r"\$assert\b",
            r"\$assume\b",
        ]

        COVERAGE_PATTERNS = [
            r"covergroup\b",
            r"coverpoint\b",
            r"cross\b",
        ]

        def _detect_frameworks(source: str) -> List[str]:
            found = []
            for fw, pats in FW_PATTERNS.items():
                if any(re.search(p, source, re.IGNORECASE) for p in pats):
                    found.append(fw)
            return found

        def _count_matches(source: str, patterns: List[str]) -> int:
            return sum(len(list(re.finditer(p, source, re.IGNORECASE))) for p in patterns)

        # Scan files
        rtl_files: List[Dict[str, Any]] = []
        tb_files: List[Dict[str, Any]] = []
        frameworks_global: Set[str] = set()
        assertion_count = 0
        coverage_count = 0

        print(f"DEBUG verification_coverage: Starting analysis of {len(v_files)} files")

        for entry in v_files:
            source = entry.get("source") or ""
            if not source.strip():
                continue

            rel = _rel_path(entry)

            if _likely_testbench_file(entry):
                tb_files.append(entry)
                print(f"DEBUG verification_coverage: Found testbench: {rel}")

                fws = _detect_frameworks(source)
                for fw in fws:
                    frameworks_global.add(fw)

                assertion_count += _count_matches(source, ASSERTION_PATTERNS)
                coverage_count += _count_matches(source, COVERAGE_PATTERNS)
            elif _is_rtl_file(entry):
                rtl_files.append(entry)
                print(f"DEBUG verification_coverage: Found RTL file: {rel}")

        print(
            f"DEBUG verification_coverage: Found {len(rtl_files)} RTL files, "
            f"{len(tb_files)} testbench files"
        )
        print(f"DEBUG verification_coverage: Frameworks: {sorted(frameworks_global)}")

        # Calculate metrics
        total_rtl = len(rtl_files)
        total_tb = len(tb_files)
        tb_ratio = (total_tb / total_rtl) if total_rtl > 0 else 0.0

        # Scoring
        def bucket_score(r: float) -> int:
            if r >= 0.8:
                return 100
            if r >= 0.6:
                return 85
            if r >= 0.4:
                return 70
            if r >= 0.2:
                return 50
            return 0 if r == 0 else 30

        s_tb_ratio = bucket_score(tb_ratio)

        # Framework score
        s_framework = 100 if frameworks_global else (40 if total_tb > 0 else 0)

        # Assertion score
        avg_assertions = (assertion_count / max(1, total_tb)) if total_tb > 0 else 0
        if avg_assertions >= 5:
            s_assert = 100
        elif avg_assertions >= 2:
            s_assert = 80
        elif avg_assertions > 0:
            s_assert = 50
        else:
            s_assert = 0 if total_tb == 0 else 20

        # Coverage score
        s_cov = 100 if coverage_count > 0 else 0

        # Tool integration
        s_tool = 50 if total_tb > 0 else 0

        score = (
            s_tb_ratio * W["testbench_ratio"]
            + s_framework * W["frameworks"]
            + s_assert * W["assertions"]
            + s_cov * W["coverage"]
            + s_tool * W["tools"]
        )

        issues: List[str] = []

        if total_tb == 0:
            issues.append("No testbench/verification files detected")
            score = 0.0
        elif tb_ratio < 0.3:
            issues.append(f"Low testbench ratio: {tb_ratio:.1%}")

        if not frameworks_global and total_tb > 0:
            issues.append("No recognized verification framework detected (UVM, SVA, Cocotb)")

        if total_tb > 0 and assertion_count == 0:
            issues.append("No SystemVerilog assertions detected in testbenches")

        if coverage_count == 0 and total_tb > 0:
            issues.append("No covergroup declarations found")

        if score >= 80 and total_tb > 0:
            issues.append("Excellent verification setup!")
        elif score >= 60 and total_tb > 0:
            issues.append("Good verification foundation - consider expanding coverage")

        score = max(0.0, min(100.0, score))
        grade = self._score_to_grade(score)

        print(f"DEBUG verification_coverage: Final score: {score:.1f}")

        metrics = {
            "total_rtl_files": total_rtl,
            "total_testbench_files": total_tb,
            "testbench_ratio": round(tb_ratio, 3),
            "frameworks_detected": sorted(list(frameworks_global)),
            "assertion_count": assertion_count,
            "coverage_declarations": coverage_count,
            "avg_assertions_per_tb": round(assertion_count / max(1, total_tb), 2),
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
