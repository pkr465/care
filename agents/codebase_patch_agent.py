"""
CARE — Codebase Analysis & Refactor Engine
Codebase Patch Agent

Analyses patches (unified diffs) against source files to identify issues
introduced by the patch.  Uses :class:`CodebaseLLMAgent` and optionally
:class:`StaticAnalyzerAgent` to compare original vs. patched code, then
writes findings to a ``patch_<filename>`` tab in
``detailed_code_review.xlsx``.

Usage:
    agent = CodebasePatchAgent(
        file_path="/path/to/original.c",
        patch_file="/path/to/change.patch",
        output_dir="./out",
    )
    result = agent.run_analysis(excel_path="out/detailed_code_review.xlsx")
"""

import logging
import os
import re
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from agents.codebase_llm_agent import CodebaseLLMAgent
    LLM_AGENT_AVAILABLE = True
except ImportError:
    LLM_AGENT_AVAILABLE = False
    CodebaseLLMAgent = None
    logger.warning("CodebaseLLMAgent not available — LLM analysis disabled for patch agent")

try:
    from agents.codebase_static_agent import StaticAnalyzerAgent
    STATIC_AGENT_AVAILABLE = True
except ImportError:
    STATIC_AGENT_AVAILABLE = False
    StaticAnalyzerAgent = None

try:
    from agents.adapters import (
        ASTComplexityAdapter,
        SecurityAdapter,
        DeadCodeAdapter,
        CallGraphAdapter,
        FunctionMetricsAdapter,
    )
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

try:
    from utils.common.excel_writer import ExcelWriter, ExcelStyle
    EXCEL_WRITER_AVAILABLE = True
except ImportError:
    EXCEL_WRITER_AVAILABLE = False
    ExcelWriter = None
    ExcelStyle = None

try:
    from utils.common.llm_tools import LLMTools
    LLM_TOOLS_AVAILABLE = True
except ImportError:
    LLM_TOOLS_AVAILABLE = False
    LLMTools = None

try:
    from utils.parsers.global_config_parser import GlobalConfig
    GLOBAL_CONFIG_AVAILABLE = True
except ImportError:
    GLOBAL_CONFIG_AVAILABLE = False
    GlobalConfig = None

try:
    from dependency_builder.config import DependencyBuilderConfig
    DEP_CONFIG_AVAILABLE = True
except ImportError:
    DEP_CONFIG_AVAILABLE = False
    DependencyBuilderConfig = None

# HITL support (optional)
try:
    from hitl import HITLContext, HITL_AVAILABLE
except ImportError:
    HITLContext = None
    HITL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatchHunk:
    """Represents a single hunk from a unified diff."""

    orig_start: int
    orig_count: int
    new_start: int
    new_count: int
    header: str = ""
    removed_lines: List[str] = field(default_factory=list)
    added_lines: List[str] = field(default_factory=list)
    context_lines: List[str] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)


@dataclass
class PatchFinding:
    """A single issue found in the patched code."""

    file_path: str
    line_number: int
    severity: str
    category: str
    description: str
    code_before: str = ""
    code_after: str = ""
    introduced_by_patch: bool = True
    issue_source: str = "patch"


# ---------------------------------------------------------------------------
# Patch Agent
# ---------------------------------------------------------------------------

class CodebasePatchAgent:
    """Analyse patches against source files to identify introduced issues.

    Workflow:

    1. Read the original source file.
    2. Parse the unified diff via :meth:`_parse_patch`.
    3. Apply the patch to produce a patched source string.
    4. Run :class:`CodebaseLLMAgent` on **both** original and patched files.
    5. Optionally run static analysis adapters on the patched file.
    6. Diff findings to isolate issues **introduced** by the patch.
    7. Write a ``patch_<filename>`` tab to ``detailed_code_review.xlsx``.

    Parameters
    ----------
    file_path : str
        Path to the **original** source file being patched.
    patch_file : str
        Path to the ``.patch`` / ``.diff`` file (unified diff format).
    output_dir : str
        Output directory for artefacts.
    config : GlobalConfig, optional
        Hierarchical configuration from ``global_config.yaml``.
    llm_tools : LLMTools, optional
        Pre-initialised LLM tools instance (dependency injection).
    dep_config : DependencyBuilderConfig, optional
        CCLS / dependency builder configuration.
    hitl_context : HITLContext, optional
        Human-in-the-loop feedback context.
    enable_adapters : bool
        Run deep static analysis adapters on the patched file.
    verbose : bool
        Enable verbose logging.
    """

    def __init__(
        self,
        file_path: str,
        patch_file: str,
        output_dir: str = "./out",
        config: Optional[Any] = None,
        llm_tools: Optional[Any] = None,
        dep_config: Optional[Any] = None,
        hitl_context: Optional[Any] = None,
        enable_adapters: bool = False,
        verbose: bool = False,
    ) -> None:
        self.file_path = Path(file_path).resolve()
        self.patch_file = Path(patch_file).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config = config
        self.llm_tools = llm_tools
        self.dep_config = dep_config
        self.hitl_context = hitl_context
        self.enable_adapters = enable_adapters
        self.verbose = verbose

        # Derive useful names
        self.filename = self.file_path.name
        self.filename_stem = self.file_path.stem

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Temp directory for analysis artefacts
        self._temp_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        excel_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full patch analysis pipeline.

        Parameters
        ----------
        excel_path : str, optional
            Path to ``detailed_code_review.xlsx`` to update with the
            ``patch_<filename>`` tab. If *None*, creates a new file in
            *output_dir*.

        Returns
        -------
        dict
            Analysis result with keys: ``status``, ``findings``,
            ``original_issue_count``, ``patched_issue_count``,
            ``new_issue_count``, ``excel_path``.
        """
        logger.info("Patch Agent: analysing %s with %s", self.filename, self.patch_file.name)

        # -- Validate inputs ------------------------------------------------
        if not self.file_path.exists():
            logger.error("Source file not found: %s", self.file_path)
            return {"status": "error", "message": f"Source file not found: {self.file_path}"}

        if not self.patch_file.exists():
            logger.error("Patch file not found: %s", self.patch_file)
            return {"status": "error", "message": f"Patch file not found: {self.patch_file}"}

        try:
            original_content = self.file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.error("Failed to read source file: %s", exc)
            return {"status": "error", "message": str(exc)}

        try:
            patch_content = self.patch_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.error("Failed to read patch file: %s", exc)
            return {"status": "error", "message": str(exc)}

        # -- Parse and apply patch ------------------------------------------
        hunks = self._parse_patch(patch_content)
        if not hunks:
            logger.warning("No hunks found in patch file")
            return {"status": "warning", "message": "No hunks found in patch", "findings": []}

        logger.info("  Parsed %d hunk(s) from patch", len(hunks))

        patched_content = self._apply_patch(original_content, hunks)

        # -- Setup temp directories -----------------------------------------
        self._temp_dir = Path(tempfile.mkdtemp(prefix="care_patch_"))
        try:
            return self._run_pipeline(
                original_content, patched_content, hunks, excel_path
            )
        finally:
            # Cleanup temp dir
            if self._temp_dir and self._temp_dir.exists():
                try:
                    shutil.rmtree(str(self._temp_dir))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        original_content: str,
        patched_content: str,
        hunks: List[PatchHunk],
        excel_path: Optional[str],
    ) -> Dict[str, Any]:
        """Run the full analysis pipeline."""

        # Write temp files for analysis
        orig_dir = self._temp_dir / "original"
        patched_dir = self._temp_dir / "patched"
        orig_dir.mkdir()
        patched_dir.mkdir()

        orig_file = orig_dir / self.filename
        patched_file = patched_dir / self.filename
        orig_file.write_text(original_content, encoding="utf-8")
        patched_file.write_text(patched_content, encoding="utf-8")

        # -- Run LLM analysis on both versions ------------------------------
        original_issues: List[Dict] = []
        patched_issues: List[Dict] = []

        if LLM_AGENT_AVAILABLE:
            logger.info("  Running LLM analysis on original file...")
            original_issues = self._run_llm_analysis(
                str(orig_dir), self.filename, "original"
            )
            logger.info("  Original: %d issue(s) found", len(original_issues))

            logger.info("  Running LLM analysis on patched file...")
            patched_issues = self._run_llm_analysis(
                str(patched_dir), self.filename, "patched"
            )
            logger.info("  Patched: %d issue(s) found", len(patched_issues))
        else:
            logger.warning("  CodebaseLLMAgent not available — skipping LLM analysis")

        # -- Run static adapters on patched file (optional) -----------------
        static_issues: List[Dict] = []
        if self.enable_adapters and ADAPTERS_AVAILABLE:
            logger.info("  Running static adapters on patched file...")
            static_issues = self._run_static_analysis(str(patched_dir), self.filename)
            logger.info("  Static analysis: %d issue(s) found", len(static_issues))

        # Merge patched_issues + static_issues
        all_patched_issues = patched_issues + static_issues

        # -- Diff findings --------------------------------------------------
        new_findings = self._diff_findings(original_issues, all_patched_issues, hunks)
        logger.info("  New issues introduced by patch: %d", len(new_findings))

        # -- Write to Excel -------------------------------------------------
        final_excel = excel_path or str(self.output_dir / "detailed_code_review.xlsx")
        self._update_excel(final_excel, new_findings)

        return {
            "status": "success",
            "filename": self.filename,
            "patch_file": str(self.patch_file),
            "original_issue_count": len(original_issues),
            "patched_issue_count": len(all_patched_issues),
            "new_issue_count": len(new_findings),
            "findings": [self._finding_to_dict(f) for f in new_findings],
            "excel_path": final_excel,
            "hunks_parsed": len(hunks),
        }

    # ------------------------------------------------------------------
    # Patch parsing
    # ------------------------------------------------------------------

    _HUNK_RE = re.compile(
        r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$"
    )

    def _parse_patch(self, patch_text: str) -> List[PatchHunk]:
        """Parse a unified diff into a list of :class:`PatchHunk` objects."""
        hunks: List[PatchHunk] = []
        current_hunk: Optional[PatchHunk] = None

        for line in patch_text.splitlines():
            m = self._HUNK_RE.match(line)
            if m:
                # Save the previous hunk
                if current_hunk is not None:
                    hunks.append(current_hunk)

                current_hunk = PatchHunk(
                    orig_start=int(m.group(1)),
                    orig_count=int(m.group(2) or 1),
                    new_start=int(m.group(3)),
                    new_count=int(m.group(4) or 1),
                    header=m.group(5).strip(),
                )
                continue

            if current_hunk is None:
                # Header lines (---, +++, diff --git, etc.) — skip
                continue

            current_hunk.raw_lines.append(line)

            if line.startswith("-"):
                current_hunk.removed_lines.append(line[1:])
            elif line.startswith("+"):
                current_hunk.added_lines.append(line[1:])
            elif line.startswith(" ") or line == "":
                current_hunk.context_lines.append(line[1:] if line.startswith(" ") else line)

        # Don't forget the last hunk
        if current_hunk is not None:
            hunks.append(current_hunk)

        return hunks

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------

    def _apply_patch(self, source: str, hunks: List[PatchHunk]) -> str:
        """Apply parsed hunks to the original source to reconstruct patched content.

        Uses a line-based approach: replaces each hunk's original region
        with the new region in order, adjusting offsets as we go.
        """
        lines = source.splitlines(keepends=True)
        offset = 0  # cumulative line offset from previous hunk applications

        for hunk in hunks:
            # Convert to 0-based indexing
            start = hunk.orig_start - 1 + offset
            end = start + hunk.orig_count

            # Build replacement lines from the raw diff
            new_lines: List[str] = []
            for raw_line in hunk.raw_lines:
                if raw_line.startswith("+"):
                    new_lines.append(raw_line[1:] + "\n")
                elif raw_line.startswith(" ") or raw_line == "":
                    content = raw_line[1:] if raw_line.startswith(" ") else raw_line
                    new_lines.append(content + "\n")
                # Lines starting with "-" are removed (not added to new_lines)

            # Replace the region
            lines[start:end] = new_lines

            # Update offset for next hunk
            offset += len(new_lines) - hunk.orig_count

        return "".join(lines)

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _run_llm_analysis(
        self, temp_dir: str, filename: str, label: str
    ) -> List[Dict]:
        """Run CodebaseLLMAgent on a single file in a temp directory.

        Returns a list of issue dicts extracted from the agent's results.
        """
        if not LLM_AGENT_AVAILABLE:
            return []

        try:
            analysis_out = os.path.join(self._temp_dir, f"llm_{label}")
            os.makedirs(analysis_out, exist_ok=True)

            agent = CodebaseLLMAgent(
                codebase_path=temp_dir,
                output_dir=analysis_out,
                config=self.config,
                llm_tools=self.llm_tools,
                max_files=1,
                file_to_fix=filename,
                hitl_context=self.hitl_context,
            )

            output_filename = f"patch_{label}_{self.filename_stem}.xlsx"
            report_path = agent.run_analysis(output_filename=output_filename)

            # Extract issues from the generated Excel
            return self._extract_issues_from_excel(report_path, label)
        except Exception as exc:
            logger.warning("LLM analysis (%s) failed: %s", label, exc)
            if self.verbose:
                logger.exception("Full traceback:")
            return []

    def _extract_issues_from_excel(
        self, excel_path: str, label: str
    ) -> List[Dict]:
        """Extract issues from a generated Excel report."""
        issues: List[Dict] = []

        if not excel_path or not Path(excel_path).exists():
            return issues

        try:
            import pandas as pd

            # Try reading the Analysis sheet
            try:
                df = pd.read_excel(excel_path, sheet_name="Analysis", header=0)
            except Exception:
                # Try first sheet if Analysis doesn't exist
                df = pd.read_excel(excel_path, header=0)

            df.columns = [str(c).strip() for c in df.columns]

            for _, row in df.iterrows():
                file_val = row.get("File") or row.get("file") or row.get("file_path")
                if pd.isna(file_val) if file_val is not None else True:
                    continue

                issue = {
                    "file_path": str(file_val),
                    "line_number": int(row.get("Line", 0)) if pd.notna(row.get("Line", None)) else 0,
                    "severity": str(row.get("Severity", "medium")),
                    "category": str(row.get("Issue_Type", row.get("Category", ""))),
                    "description": str(row.get("Description", row.get("Code", ""))),
                    "code": str(row.get("Code", "")),
                    "fixed_code": str(row.get("Fixed_Code", "")),
                    "source": label,
                }
                issues.append(issue)

        except ImportError:
            logger.warning("pandas not available — cannot extract issues from Excel")
        except Exception as exc:
            logger.warning("Failed to extract issues from %s: %s", excel_path, exc)

        return issues

    # ------------------------------------------------------------------
    # Static analysis
    # ------------------------------------------------------------------

    def _run_static_analysis(
        self, temp_dir: str, filename: str
    ) -> List[Dict]:
        """Run static analysis adapters on the patched file."""
        if not ADAPTERS_AVAILABLE:
            return []

        issues: List[Dict] = []
        try:
            # Build a minimal file cache for adapters
            from agents.core.file_processor import FileProcessor

            processor = FileProcessor(
                codebase_path=temp_dir,
                exclude_dirs=[],
            )
            file_cache = processor.process_files()

            adapters = [
                ("ast_complexity", ASTComplexityAdapter()),
                ("security", SecurityAdapter()),
            ]

            for name, adapter in adapters:
                try:
                    result = adapter.analyze(
                        file_cache, ccls_navigator=None, dependency_graph={}
                    )
                    if result.get("tool_available", False):
                        # Extract issues from adapter results
                        for finding in result.get("findings", result.get("issues", [])):
                            issues.append({
                                "file_path": finding.get("file", filename),
                                "line_number": finding.get("line", 0),
                                "severity": finding.get("severity", "medium"),
                                "category": f"static_{name}",
                                "description": finding.get("description", finding.get("message", "")),
                                "code": finding.get("code", ""),
                                "source": f"static_{name}",
                            })
                except Exception as exc:
                    logger.warning("Adapter %s failed: %s", name, exc)

        except ImportError:
            logger.warning("FileProcessor not available — skipping static analysis")
        except Exception as exc:
            logger.warning("Static analysis failed: %s", exc)

        return issues

    # ------------------------------------------------------------------
    # Findings diff
    # ------------------------------------------------------------------

    @staticmethod
    def _fingerprint_issue(issue: Dict) -> str:
        """Create a fingerprint for an issue to enable deduplication.

        Uses: (filename, line_range_bucket, category, description_prefix).
        Line numbers are bucketed into ranges of 5 to handle minor drift.
        """
        filename = Path(issue.get("file_path", "")).name
        line = issue.get("line_number", 0)
        line_bucket = (line // 5) * 5  # bucket into groups of 5
        category = issue.get("category", "").lower().strip()
        desc = issue.get("description", "")[:80].lower().strip()

        return f"{filename}|{line_bucket}|{category}|{desc}"

    def _diff_findings(
        self,
        original_issues: List[Dict],
        patched_issues: List[Dict],
        hunks: List[PatchHunk],
    ) -> List[PatchFinding]:
        """Identify issues that are NEW in the patched version.

        An issue is considered 'new' if it:
        1. Was NOT present in the original (by fingerprint), OR
        2. Falls within or near a hunk's modified line range.
        """
        # Build fingerprint set from original
        orig_fingerprints = {self._fingerprint_issue(i) for i in original_issues}

        # Build a set of line ranges modified by hunks
        modified_ranges: List[Tuple[int, int]] = []
        for hunk in hunks:
            start = hunk.new_start
            end = start + hunk.new_count
            modified_ranges.append((start, end))

        def _in_modified_range(line: int) -> bool:
            """Check if a line falls within or near a modified hunk range."""
            for start, end in modified_ranges:
                if (start - 3) <= line <= (end + 3):
                    return True
            return False

        new_findings: List[PatchFinding] = []

        for issue in patched_issues:
            fp = self._fingerprint_issue(issue)
            line_num = issue.get("line_number", 0)

            # Issue is new if not in original OR in a modified range
            if fp not in orig_fingerprints or _in_modified_range(line_num):
                finding = PatchFinding(
                    file_path=issue.get("file_path", self.filename),
                    line_number=line_num,
                    severity=issue.get("severity", "medium"),
                    category=issue.get("category", ""),
                    description=issue.get("description", ""),
                    code_before=issue.get("code", ""),
                    code_after=issue.get("fixed_code", ""),
                    introduced_by_patch=True,
                    issue_source=issue.get("source", "patch"),
                )
                new_findings.append(finding)

        return new_findings

    # ------------------------------------------------------------------
    # Excel output
    # ------------------------------------------------------------------

    def _update_excel(
        self, excel_path: str, findings: List[PatchFinding]
    ) -> None:
        """Write patch findings to a ``patch_<filename>`` tab in the Excel file."""
        if not EXCEL_WRITER_AVAILABLE:
            logger.warning("ExcelWriter not available — writing findings as JSON instead")
            self._write_findings_json(findings)
            return

        sheet_name = f"patch_{self.filename_stem}"
        # Truncate sheet name to Excel's 31-char limit
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]

        headers = [
            "File",
            "Line",
            "Severity",
            "Category",
            "Description",
            "Code_Before",
            "Code_After",
            "Introduced_By_Patch",
            "Issue_Source",
        ]

        data_rows: List[List[Any]] = []
        for f in findings:
            data_rows.append([
                f.file_path,
                f.line_number,
                f.severity,
                f.category,
                f.description,
                f.code_before,
                f.code_after,
                "YES" if f.introduced_by_patch else "NO",
                f.issue_source,
            ])

        try:
            # Try to open existing workbook and add a new sheet
            excel_file = Path(excel_path)

            if excel_file.exists():
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(str(excel_file))

                    # Remove existing sheet with the same name if present
                    if sheet_name in wb.sheetnames:
                        del wb[sheet_name]

                    ws = wb.create_sheet(title=sheet_name)

                    # Write header row
                    for col_idx, header in enumerate(headers, 1):
                        ws.cell(row=1, column=col_idx, value=header)

                    # Write data rows
                    for row_idx, row_data in enumerate(data_rows, 2):
                        for col_idx, value in enumerate(row_data, 1):
                            ws.cell(row=row_idx, column=col_idx, value=value)

                    # Auto-fit columns
                    for col_idx, header in enumerate(headers, 1):
                        max_len = len(header)
                        for row_data in data_rows:
                            if col_idx <= len(row_data):
                                val_len = len(str(row_data[col_idx - 1]))
                                max_len = max(max_len, min(val_len, 60))
                        ws.column_dimensions[
                            openpyxl.utils.get_column_letter(col_idx)
                        ].width = max_len + 4

                    wb.save(str(excel_file))
                    logger.info(
                        "Updated %s with '%s' tab (%d findings)",
                        excel_path, sheet_name, len(findings),
                    )
                    return

                except Exception as exc:
                    logger.warning("Failed to update existing Excel: %s — creating new", exc)

            # Create new workbook with ExcelWriter
            writer = ExcelWriter(str(excel_path))
            writer.add_table_sheet(
                headers=headers,
                data_rows=data_rows,
                sheet_name=sheet_name,
                status_column="Severity",
            )
            writer.save()
            logger.info(
                "Created %s with '%s' tab (%d findings)",
                excel_path, sheet_name, len(findings),
            )

        except Exception as exc:
            logger.error("Failed to write Excel: %s", exc)
            self._write_findings_json(findings)

    def _write_findings_json(self, findings: List[PatchFinding]) -> None:
        """Fallback: write findings as JSON."""
        import json

        json_path = self.output_dir / f"patch_{self.filename_stem}_findings.json"
        data = [self._finding_to_dict(f) for f in findings]
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, default=str)
        logger.info("Findings written to %s", json_path)

    @staticmethod
    def _finding_to_dict(finding: PatchFinding) -> Dict[str, Any]:
        """Convert a PatchFinding to a plain dict."""
        return {
            "file_path": finding.file_path,
            "line_number": finding.line_number,
            "severity": finding.severity,
            "category": finding.category,
            "description": finding.description,
            "code_before": finding.code_before,
            "code_after": finding.code_after,
            "introduced_by_patch": finding.introduced_by_patch,
            "issue_source": finding.issue_source,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_patch_summary(self) -> Dict[str, Any]:
        """Parse the patch and return a summary without running full analysis."""
        if not self.patch_file.exists():
            return {"error": "Patch file not found"}

        patch_content = self.patch_file.read_text(encoding="utf-8", errors="ignore")
        hunks = self._parse_patch(patch_content)

        total_added = sum(len(h.added_lines) for h in hunks)
        total_removed = sum(len(h.removed_lines) for h in hunks)

        return {
            "patch_file": str(self.patch_file),
            "target_file": str(self.file_path),
            "hunk_count": len(hunks),
            "lines_added": total_added,
            "lines_removed": total_removed,
            "net_change": total_added - total_removed,
            "hunks": [
                {
                    "header": h.header,
                    "orig_range": f"{h.orig_start},{h.orig_count}",
                    "new_range": f"{h.new_start},{h.new_count}",
                    "added": len(h.added_lines),
                    "removed": len(h.removed_lines),
                }
                for h in hunks
            ],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="CARE Codebase Patch Agent — Analyse patches for introduced issues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--file-path",
        required=True,
        help="Path to the original source file",
    )
    parser.add_argument(
        "--patch-file",
        required=True,
        help="Path to the .patch/.diff file (unified diff format)",
    )
    parser.add_argument(
        "--excel-path",
        default=None,
        help="Path to detailed_code_review.xlsx to update",
    )
    parser.add_argument(
        "-d", "--out-dir",
        default="./out",
        help="Output directory",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to global_config.yaml",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model in provider::model format",
    )
    parser.add_argument(
        "--enable-adapters",
        action="store_true",
        default=False,
        help="Run deep static analysis adapters on patched file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = None
    if GLOBAL_CONFIG_AVAILABLE and args.config_file:
        try:
            config = GlobalConfig(args.config_file)
        except Exception as e:
            logger.warning("Could not load config: %s", e)

    # Setup LLM tools
    llm_tools = None
    if LLM_TOOLS_AVAILABLE:
        try:
            if args.llm_model:
                llm_tools = LLMTools(model=args.llm_model)
            elif config:
                model_str = config.get("llm.model")
                llm_tools = LLMTools(model=model_str) if model_str else LLMTools()
            else:
                llm_tools = LLMTools()
        except Exception as e:
            logger.warning("Could not initialise LLMTools: %s", e)

    # Run agent
    agent = CodebasePatchAgent(
        file_path=args.file_path,
        patch_file=args.patch_file,
        output_dir=args.out_dir,
        config=config,
        llm_tools=llm_tools,
        enable_adapters=args.enable_adapters,
        verbose=args.verbose,
    )

    result = agent.run_analysis(excel_path=args.excel_path)

    print(f"\n{'='*60}")
    print(f" Patch Analysis Results: {agent.filename}")
    print(f"{'='*60}")
    print(f"  Status:           {result.get('status')}")
    print(f"  Hunks parsed:     {result.get('hunks_parsed', 0)}")
    print(f"  Original issues:  {result.get('original_issue_count', 0)}")
    print(f"  Patched issues:   {result.get('patched_issue_count', 0)}")
    print(f"  NEW issues:       {result.get('new_issue_count', 0)}")
    print(f"  Excel output:     {result.get('excel_path', 'N/A')}")
    print(f"{'='*60}")

    if result.get("findings"):
        print(f"\n  Findings:")
        for i, f in enumerate(result["findings"], 1):
            print(f"    {i}. [{f['severity']}] {f['category']} — {f['description'][:80]}")
            print(f"       Line {f['line_number']} in {f['file_path']}")

    sys.exit(0 if result.get("status") == "success" else 1)
