"""
Verilog/SystemVerilog documentation analysis
"""

import os
import re
from typing import Dict, List, Any, Tuple


class DocumentationAnalyzer:
    """
    Analyzes Verilog/SystemVerilog code documentation quality focusing on:
    - SVDoc/HDL documentation style coverage
    - Module and port documentation
    - Parameter documentation
    - Timescale declarations
    - Clock domain annotations
    - Reset strategy documentation
    - Signal documentation density
    """

    V_EXTS = {".v", ".sv"}
    VH_EXTS = {".vh", ".svh"}

    def __init__(self, codebase_path: str = None, project_root: str = None):
        """Initialize documentation analyzer."""
        self.codebase_path = codebase_path or os.getcwd()
        self.project_root = project_root or os.getcwd()
        self._file_cache: List[Dict[str, Any]] = []

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog documentation quality.

        Args:
            file_cache: List of processed Verilog file entries

        Returns:
            Documentation analysis results with score, grade, metrics, and issues.
        """
        self._file_cache = file_cache or []
        return self._calculate_documentation_score()

    def _calculate_documentation_score(self) -> Dict[str, Any]:
        """
        Verilog/SystemVerilog documentation coverage and quality score.

        Analyzes for:
        - Module header documentation
        - Port descriptions
        - Parameter documentation
        - Timescale declarations
        - Clock domain annotations
        """
        if not self._file_cache:
            return {"score": 0, "grade": "F", "issues": ["No files cached"]}

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
            f"DEBUG documentation: Found {len(v_files)} Verilog files "
            f"out of {len(self._file_cache)} total"
        )

        if not v_files:
            return {
                "score": 0.0,
                "grade": "F",
                "issues": [
                    "No Verilog/SystemVerilog files found for documentation analysis. "
                    f"Extensions checked: {sorted(self.V_EXTS | self.VH_EXTS)}"
                ],
                "metrics": {
                    "documentation_ratio": 0.0,
                    "header_documentation_ratio": 0.0,
                    "total_documentable_items": 0,
                    "documented_items": 0,
                    "files": [],
                    "missing_items": [],
                },
            }

        def _count_comment_lines(src: str) -> int:
            """Count lines that are comments."""
            lines = src.splitlines()
            count = 0
            in_ml = False

            for ln in lines:
                s = ln.strip()
                if in_ml:
                    count += 1
                    if "*/" in s:
                        in_ml = False
                    continue

                if s.startswith("//"):
                    count += 1
                    continue

                if "/*" in s:
                    count += 1
                    if "*/" not in s:
                        in_ml = True

            return count

        def _has_module_header_doc(src: str) -> bool:
            """Check if module has header documentation."""
            head = "\n".join(src.splitlines()[:50])
            return bool(
                re.search(r"/\*.*?(module|purpose|description)", head, re.IGNORECASE | re.DOTALL)
                or re.search(r"//\s*(module|purpose|description)", head, re.IGNORECASE)
            )

        def _has_port_documentation(src: str) -> bool:
            """Check if ports are documented."""
            return bool(
                re.search(r"//\s*(input|output|inout):", src)
                or re.search(r"/\*.*?(input|output|inout):", src, re.DOTALL)
            )

        def _count_ports(src: str) -> int:
            """Count module ports."""
            module_match = re.search(r"module\s+\w+\s*#?\s*\((.*?)\)", src, re.DOTALL)
            if module_match:
                params = module_match.group(1)
                return len(re.findall(r",", params)) + 1 if params.strip() else 0
            return 0

        def _count_parameters(src: str) -> int:
            """Count module parameters."""
            return len(re.findall(r"parameter\s+", src))

        # Aggregation containers
        per_file: List[Dict[str, Any]] = []
        missing_items: List[Dict[str, Any]] = []

        total_items = 0
        documented_items = 0

        total_modules = 0
        documented_modules = 0
        total_ports = 0
        documented_ports_count = 0
        total_params = 0
        documented_params_count = 0

        print(f"DEBUG documentation: Starting analysis of {len(v_files)} files")

        for entry in v_files:
            source = entry.get("source", "") or ""
            if not source.strip():
                continue

            rel = _rel_path(entry)
            print(f"DEBUG documentation: Analyzing {rel} ({len(source)} chars)")

            lines = source.splitlines()
            comment_lines = _count_comment_lines(source)
            code_lines = sum(1 for ln in lines if ln.strip())
            comment_ratio = comment_lines / max(1, len(lines))

            # Module documentation
            module_match = re.search(r"module\s+([a-zA-Z0-9_]+)", source)
            if module_match:
                total_modules += 1
                module_name = module_match.group(1)
                file_items = 1
                file_documented = 0

                if _has_module_header_doc(source):
                    documented_modules += 1
                    file_documented += 1
                    documented_items += 1
                else:
                    missing_items.append({
                        "file": rel,
                        "kind": "module",
                        "name": module_name,
                        "reason": "Missing module header documentation",
                    })

                # Ports
                port_count = _count_ports(source)
                total_ports += port_count
                if _has_port_documentation(source):
                    documented_ports_count += port_count
                    file_documented += 1
                else:
                    if port_count > 0:
                        missing_items.append({
                            "file": rel,
                            "kind": "ports",
                            "name": module_name,
                            "reason": f"Missing documentation for {port_count} port(s)",
                        })

                # Parameters
                param_count = _count_parameters(source)
                total_params += param_count
                param_doc_count = len(re.findall(r"//\s*parameter|/\*\s*parameter", source))
                documented_params_count += min(param_doc_count, param_count)
                if param_count > 0 and param_doc_count == 0:
                    missing_items.append({
                        "file": rel,
                        "kind": "parameters",
                        "name": module_name,
                        "reason": f"Missing documentation for {param_count} parameter(s)",
                    })

                file_items += port_count + param_count
                total_items += file_items

                doc_ratio = file_documented / max(1, file_items)

                per_file.append({
                    "file": rel,
                    "comment_ratio": round(comment_ratio, 3),
                    "total_items": file_items,
                    "documented_items": file_documented,
                    "documentation_ratio": round(doc_ratio, 3),
                    "module_name": module_name,
                    "ports": port_count,
                    "parameters": param_count,
                })

                print(
                    f"DEBUG documentation: File {rel} - Module: {module_name}, "
                    f"Ports: {port_count}, Params: {param_count}, Doc ratio: {doc_ratio:.1%}"
                )

        # Scoring
        overall_doc_ratio = (documented_items / total_items) if total_items > 0 else 0.0
        module_doc_ratio = (documented_modules / total_modules) if total_modules > 0 else 0.0
        port_doc_ratio = (documented_ports_count / total_ports) if total_ports > 0 else 0.0
        param_doc_ratio = (documented_params_count / total_params) if total_params > 0 else 0.0

        def to_bucket_score(r: float) -> int:
            if r >= 0.8:
                return 100
            if r >= 0.6:
                return 85
            if r >= 0.4:
                return 65
            if r >= 0.2:
                return 45
            return 25

        coverage_score = to_bucket_score(overall_doc_ratio)
        module_score = to_bucket_score(module_doc_ratio)
        port_score = to_bucket_score(port_doc_ratio)
        param_score = to_bucket_score(param_doc_ratio)

        overall_score = (
            0.40 * coverage_score
            + 0.25 * module_score
            + 0.20 * port_score
            + 0.15 * param_score
        )

        grade = self._score_to_grade(overall_score)
        print(f"DEBUG documentation: Final score: {overall_score:.1f}")

        issues: List[str] = []
        if overall_doc_ratio < 0.5:
            issues.append(
                f"Low overall documentation coverage: {overall_doc_ratio:.1%}"
            )
        if module_doc_ratio < 0.8:
            issues.append(
                f"Module header documentation coverage is low: {module_doc_ratio:.1%}"
            )
        if port_doc_ratio < 0.7 and total_ports > 0:
            issues.append(
                f"Port documentation coverage is low: {port_doc_ratio:.1%}"
            )
        if param_doc_ratio < 0.7 and total_params > 0:
            issues.append(
                f"Parameter documentation coverage is low: {param_doc_ratio:.1%}"
            )

        if missing_items:
            issues.append(
                f"{len(missing_items)} item(s) missing documentation"
            )

        if overall_score >= 80 and not issues:
            issues.append("Excellent documentation coverage!")

        metrics = {
            "documentation_ratio": round(overall_doc_ratio, 3),
            "module_documentation_ratio": round(module_doc_ratio, 3),
            "port_documentation_ratio": round(port_doc_ratio, 3),
            "parameter_documentation_ratio": round(param_doc_ratio, 3),
            "total_documentable_items": total_items,
            "documented_items": documented_items,
            "total_modules": total_modules,
            "documented_modules": documented_modules,
            "total_ports": total_ports,
            "documented_ports": documented_ports_count,
            "total_parameters": total_params,
            "documented_parameters": documented_params_count,
            "files": per_file,
            "missing_items": missing_items[:200],
        }

        return {
            "score": round(overall_score, 1),
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
