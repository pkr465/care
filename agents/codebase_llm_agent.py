import os
import re
import json
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# --- Imports ---
from utils.common.llm_tools import LLMTools

# --- New Ingestion & Dependency Services ---
try:
    # Assumes these files are accessible in the python path
    from dependency_builder.ccls_ingestion import CCLSIngestion
    from dependency_builder.dependency_service import DependencyService
    DEPENDENCY_SERVICES_AVAILABLE = True
except ImportError:
    print("Warning: Ingestion/Dependency services not found. Running in standalone heuristic mode.")
    CCLSIngestion = None
    DependencyService = None
    DEPENDENCY_SERVICES_AVAILABLE = False

# Import the Email Reporter
try:
    from utils.common.email_reporter import CodebaseEmailReporter
except ImportError:
    print("Warning: CodebaseEmailReporter not found. Email functionality will be disabled.")
    CodebaseEmailReporter = None

# Try importing the prompt
try:
    from prompts.codebase_analysis_prompt import CODEBASE_ANALYSIS_PROMPT
except ImportError:
    CODEBASE_ANALYSIS_PROMPT = "Error: Prompt file not found."

class CodebaseLLMAgent:
    """
    LLM-Enhanced Agent for strict C/C++ code review.
    
    UPDATED ARCHITECTURE:
    1. Ingestion Phase: Runs CCLS indexing to build a dependency graph (Optional via --use-ccls).
    2. Semantic Chunking: Combines physical code blocks with dependency context.
    3. Anchor Logic: Maps LLM findings back to exact source lines.
    """

    C_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
    
    # Chunking Config
    TARGET_CHUNK_CHARS = 12000  # ~3k tokens
    OVERLAP_LINES = 25

    def __init__(
        self,
        codebase_path: str,
        output_dir: str = "./out",
        exclude_dirs: Optional[List[str]] = None,
        max_files: int = 10000,
        use_ccls: bool = False,
        file_to_fix: Optional[str] = None
    ):
        """
        :param codebase_path: Root of the C/C++ project.
        :param output_dir: Directory for reports and CCLS cache.
        :param use_ccls: Boolean to enable/disable CCLS dependency analysis.
        :param file_to_fix: Specific relative path to a file to analyze (ignores others).
        """
        self.codebase_path = Path(codebase_path).resolve()
        
        # --- PATH FIX: Resolve output_dir relative to codebase_path if relative ---
        # This prevents 'out/out' issues if the script is run from different directories.
        if os.path.isabs(output_dir):
            self.output_dir = output_dir
        else:
            # Normalize "./out" to just "out" and join with codebase root
            clean_out = output_dir.replace("./", "").strip("/")
            if not clean_out: clean_out = "out"
            self.output_dir = output_dir

        self.project_name = self.codebase_path.name
        self.max_files = max_files
        self.run_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.use_ccls = use_ccls 
        self.file_to_fix = file_to_fix
        
        self.exclude_dirs = set(exclude_dirs or [
            ".git", "build", "dist", ".idea", ".vscode", 
            "node_modules", "third_party", "__pycache__", ".ccls-cache"
        ])
        
        self.llm_tools = LLMTools()
        self.results: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Initialize Services based on Flag AND Availability
        if self.use_ccls:
            if DEPENDENCY_SERVICES_AVAILABLE:
                print("[*] CCLS Dependency Services ENABLED.")
                self.ingestion = CCLSIngestion()
                self.dep_service = DependencyService()
            else:
                print("[!] Warning: --use-ccls requested but dependency services are not installed. Reverting to heuristic mode.")
                self.ingestion = None
                self.dep_service = None
        else:
            print("[*] CCLS Dependency Services DISABLED (heuristic mode).")
            self.ingestion = None
            self.dep_service = None
            
        self.is_indexed = False

    def run_analysis(
        self, 
        output_filename: str = "detailed_code_review.xlsx", 
        email_recipient: str = "sendpavanr@gmail.com"
    ) -> str:
        """
        Main Execution Pipeline:
        1. Ingestion (CCLS Indexing) - Only if enabled (Indexes full repo for context)
        2. File Discovery (Handles single file target)
        3. Semantic Analysis (File -> Chunk -> Fetch Context -> LLM)
        4. Reporting (Excel + Email)
        """
        print(f"[*] Starting analysis (Run ID: {self.run_id}) on: {self.codebase_path}")
        print(f"[*] Output Directory: {self.output_dir}")
        if self.file_to_fix:
            print(f"[*] Targeted Mode: Analyzing specific file '{self.file_to_fix}'")
        
        # --- 1. Ingestion Step ---
        if self.use_ccls and self.ingestion:
            print(f"[*] Triggering CCLS Ingestion for '{self.project_name}'...")
            try:
                self.is_indexed = self.ingestion.run_indexing(
                    project_root=str(self.codebase_path),
                    output_dir=self.output_dir,
                    unique_project_prefix=self.project_name
                )
                if not self.is_indexed:
                    print("[!] Warning: Ingestion failed or timed out. Analysis will proceed without semantic context.")
            except Exception as e:
                print(f"[!] Critical Error during CCLS Ingestion: {e}")
                self.is_indexed = False
        
        # --- 2. Gather Files ---
        files = self._gather_files()
        file_count = len(files)
        
        if file_count == 0:
            print("[!] No matching files found to analyze. Exiting.")
            return ""

        print(f"[*] Found {file_count} C/C++ files to analyze.")
        
        # --- 3. Analysis Loop ---
        for i, file_path in enumerate(files):
            try:
                rel_path = str(file_path.relative_to(self.codebase_path))
            except ValueError:
                rel_path = str(file_path)

            print(f"[{i+1}/{file_count}] Analyzing: {rel_path}...")
            
            try:
                self._analyze_single_file(file_path, rel_path)
            except Exception as e:
                print(f"    ! Error analyzing {rel_path}: {e}")
                self.errors.append({"file": rel_path, "error": str(e)})

        # --- 4. Report Generation ---
        json_path = os.path.join(self.output_dir, "llm_analysis_metrics.jsonl")
        self._generate_json_metrics(json_path)
        
        # --- PATH FIX: Handle output filename intelligently ---
        if os.path.isabs(output_filename):
            full_out_path = output_filename
        elif os.path.dirname(output_filename):
            # If user provided a path with dirs (e.g. "out/report.xlsx"), use as is to prevent double nesting
            full_out_path = output_filename
        else:
            # Pure filename, join with forced output_dir
            full_out_path = os.path.join(self.output_dir, output_filename)
            
        excel_path = self._generate_excel_report(full_out_path)

        # --- 5. Email Notification ---
        if CodebaseEmailReporter:
            self._trigger_email_report(email_recipient, excel_path, file_count)
        else:
            print("[!] Skipping email report: CodebaseEmailReporter class missing.")

        return excel_path

    def _analyze_single_file(self, file_path: Path, rel_path: str):
        """
        Analyzes a single file by splitting it into Semantic Chunks.
        Fetches dependency context for each chunk to ensure high accuracy.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                code_content = f.read()
            
            if not code_content.strip():
                return

            # 1. Physical Chunking (Brace Counting)
            chunks = self._smart_chunk_code(code_content)
            total_chunks = len(chunks)
            prev_chunk_tail = ""

            for chunk_idx, (chunk_text, start_line) in enumerate(chunks):
                
                # Prepare numbered lines for the prompt
                raw_lines = chunk_text.split('\n')
                chunk_line_count = len(raw_lines)
                end_line = start_line + chunk_line_count
                
                numbered_lines = []
                for idx, line in enumerate(raw_lines):
                    numbered_lines.append(f"{start_line + idx:5d} | {line}")
                
                numbered_code_block = "\n".join(numbered_lines)

                # 2. Context Retrieval (Dependencies)
                dependency_context = ""
                # Only fetch dependencies if enabled, available, and indexed
                if self.use_ccls and self.is_indexed and self.dep_service:
                    dependency_context = self._fetch_chunk_dependencies(
                        rel_path, start_line, end_line
                    )

                # 3. Context Construction (Previous Chunk Tail + Dependencies)
                context_header = ""
                if prev_chunk_tail:
                    context_header += (
                        f"// ... [CONTEXT: Previous Lines] ...\n"
                        f"{prev_chunk_tail}\n"
                    )
                
                if dependency_context:
                    context_header += (
                        f"\n// ... [CONTEXT: External Dependencies & Definitions] ...\n"
                        f"{dependency_context}\n"
                        f"// ... [End External Context] ...\n"
                    )

                final_chunk_text = (
                    f"{context_header}\n"
                    f"// ... [CURRENT CHUNK START: {rel_path} Lines {start_line}-{end_line}] ...\n"
                    f"{numbered_code_block}"
                )

                # Update overlap for next chunk
                if len(numbered_lines) > self.OVERLAP_LINES:
                    prev_tail_lines = numbered_lines[-self.OVERLAP_LINES:]
                else:
                    prev_tail_lines = numbered_lines
                prev_chunk_tail = "\n".join(prev_tail_lines)

                if total_chunks > 1:
                    print(f"    > Processing Chunk {chunk_idx + 1}/{total_chunks} (Lines {start_line}-{end_line})...")
                
                # 4. LLM Call
                final_prompt = f"""
                {CODEBASE_ANALYSIS_PROMPT}
                
                TARGET SOURCE CODE ({rel_path} - Part {chunk_idx+1}/{total_chunks}):
                ```cpp
                {final_chunk_text}
                ```
                """
                
                response = self.llm_tools.llm_call(final_prompt)
                
                # 5. Parsing
                parsed_issues = self._parse_llm_response(
                    response, 
                    rel_path, 
                    chunk_text=chunk_text, 
                    start_line=start_line,
                    chunk_line_count=chunk_line_count
                )
                
                if parsed_issues:
                    self.results.extend(parsed_issues)

        except Exception as e:
            raise e

    def _fetch_chunk_dependencies(self, rel_path: str, start_line: int, end_line: int) -> str:
        """
        Uses DependencyService to fetch relevant definitions (structs, globals, macros)
        used within the specified line range.
        """
        try:
            # Use 'fetch_dependencies_by_file' to get context for the range
            response = self.dep_service.perform_fetch(
                project_root=str(self.codebase_path),
                output_dir=self.output_dir,
                codebase_identifier=self.project_name,
                endpoint_type="fetch_dependencies_by_file", 
                file_name=rel_path,
                start=start_line,
                end=end_line,
                level=1 
            )
            
            data = response.get("data", [])
            if not data:
                return ""

            # Format the JSON data into a C-comment style block for the LLM
            context_str = []
            if isinstance(data, list):
                for item in data[:10]: # Limit to top 10 to save tokens
                    if isinstance(item, dict):
                        name = item.get("name", "Unknown")
                        kind = item.get("kind", "Unknown")
                        snippet = item.get("snippet", "").strip()
                        file_src = item.get("file", "")
                        if snippet:
                            context_str.append(f"// ({kind}) {name} from {file_src}:\n{snippet}")
            
            return "\n\n".join(context_str)

        except Exception as e:
            # Fallback: If dependency fetch fails, log it and return empty context
            # so the main analysis doesn't crash.
            print(f"[!] Warning: Dependency fetch failed for {rel_path}: {e}")
            return ""

    def _smart_chunk_code(self, content: str) -> List[Tuple[str, int]]:
        """
        Splits code using Brace Counting logic.
        Ensures splits happen only at balance 0 (between functions).
        """
        if len(content) < self.TARGET_CHUNK_CHARS * 1.5:
            return [(content, 1)]

        chunks = []
        current_pos = 0
        total_len = len(content)
        
        last_calc_pos = 0
        last_calc_line = 1
        
        def get_line_at_index(target_index):
            nonlocal last_calc_pos, last_calc_line
            segment = content[last_calc_pos:target_index]
            newlines = segment.count('\n')
            last_calc_line += newlines
            last_calc_pos = target_index
            return last_calc_line

        while current_pos < total_len:
            start_index = current_pos
            start_line = get_line_at_index(start_index)
            
            if (total_len - start_index) < self.TARGET_CHUNK_CHARS:
                chunks.append((content[start_index:], start_line))
                break

            scan_idx = start_index
            brace_balance = 0
            found_split = False
            
            # Scan for a split point
            while scan_idx < total_len:
                char = content[scan_idx]
                if char == '{':
                    brace_balance += 1
                elif char == '}':
                    brace_balance -= 1
                
                chunk_size = scan_idx - start_index
                
                # Check for split condition: Enough chars AND balanced braces
                if chunk_size >= self.TARGET_CHUNK_CHARS and brace_balance <= 0:
                    split_idx = scan_idx + 1
                    chunks.append((content[start_index:split_idx], start_line))
                    current_pos = split_idx
                    found_split = True
                    break
                
                scan_idx += 1
            
            # Fallback if no clean split found (e.g., massive function)
            if not found_split:
                limit = min(total_len, start_index + self.TARGET_CHUNK_CHARS + 5000)
                newline_search = content.find('\n', limit)
                split_idx = newline_search + 1 if newline_search != -1 else total_len
                chunks.append((content[start_index:split_idx], start_line))
                current_pos = split_idx
                
        return chunks
        
       
    def _gather_files(self) -> List[Path]:
        """Recursively find C/C++ files, OR return the specific file_to_fix if set."""
        
        # --- Single File Logic ---
        if self.file_to_fix:
            target_path = Path(self.file_to_fix)
            
            # If path is relative, resolve it against codebase_path
            if not target_path.is_absolute():
                target_path = self.codebase_path / target_path
            
            if target_path.exists() and target_path.is_file():
                return [target_path]
            else:
                print(f"[!] Error: Target file not found: {target_path}")
                return []

        # --- Bulk Logic ---
        found_files = []
        for root, dirs, filenames in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for fname in filenames:
                path = Path(root) / fname
                if path.suffix.lower() in self.C_EXTS:
                    found_files.append(path)
                    if len(found_files) >= self.max_files:
                        return found_files
        return found_files

    def _parse_llm_response(self, response: str, file_path: str, chunk_text: str, start_line: int, chunk_line_count: int) -> List[Dict]:
        """Parses LLM output using Anchor Logic for exact line matching."""
        issues = []
        raw_blocks = response.split("---ISSUE---")
        
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
                
            issue_data = {"File": file_path}
            
            patterns = {
                "Title": r"Title:\s*(.+)",
                "Severity": r"Severity:\s*(.+)",
                "Confidence": r"Confidence:\s*(.+)",
                "Category": r"Category:\s*(.+)",
                "Description": r"Description:\s*(.+)",
                "Suggestion": r"Suggestion:\s*(.+)",
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, block, re.IGNORECASE)
                issue_data[key] = match.group(1).strip() if match else "N/A"

            code_match = re.search(r"Code:\s*```(?:\w+)?\n(.*?)\n```", block, re.DOTALL)
            if not code_match: 
                code_match = re.search(r"Code:\s*(.+?)(?=\nFixed_Code:|$)", block, re.DOTALL)
            raw_code_snippet = code_match.group(1).strip() if code_match else ""
            issue_data["Code"] = raw_code_snippet

            fixed_match = re.search(r"Fixed_Code:\s*```(?:\w+)?\n(.*?)\n```", block, re.DOTALL)
            if not fixed_match:
                fixed_match = re.search(r"Fixed_Code:\s*(.+?)(?=$)", block, re.DOTALL)
            issue_data["Fixed_Code"] = fixed_match.group(1).strip() if fixed_match else "N/A"

            # --- ANCHOR LOGIC ---
            calculated_line = 0
            found_by_anchor = False

            if raw_code_snippet:
                idx = chunk_text.find(raw_code_snippet)
                if idx == -1:
                    first_line = raw_code_snippet.split('\n')[0].strip()
                    if len(first_line) > 10: 
                        idx = chunk_text.find(first_line)

                if idx != -1:
                    newlines_before = chunk_text[:idx].count('\n')
                    calculated_line = start_line + newlines_before
                    issue_data["Line"] = str(calculated_line)
                    found_by_anchor = True

            if not found_by_anchor:
                line_match = re.search(r"Line\D*(\d+)", block, re.IGNORECASE)
                if line_match:
                    raw_val = int(line_match.group(1))
                    if raw_val < chunk_line_count and raw_val < start_line:
                        issue_data["Line"] = str(start_line + raw_val - 1)
                    else:
                        issue_data["Line"] = str(raw_val)
                else:
                    issue_data["Line"] = str(start_line)

            if issue_data["Title"] != "N/A":
                issues.append(issue_data)

        return issues

    def _generate_json_metrics(self, output_path: str):
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.results:
                agent_metric = {
                    "run_id": self.run_id,
                    "generated_at": self.start_time.isoformat(),
                    "file_path": item.get("File"),
                    "line_number": int(item.get("Line", 0)),
                    "severity": item.get("Severity"),
                    "issue_type": item.get("Category"),
                    "bad_code_snippet": item.get("Code"),
                    "suggested_fix": item.get("Fixed_Code"),
                    "rationale": item.get("Suggestion")
                }
                f.write(json.dumps(agent_metric) + '\n')

    def _generate_excel_report(self, output_path: str) -> str:
        """Generates comprehensive Excel file using XlsxWriter."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not self.results:
            print("[*] No critical issues found. Generating empty report.")
            df = pd.DataFrame(columns=[
                "S.No", "Title", "Severity", "Confidence", "Category", 
                "File", "Line", "Description", "Suggestion", 
                "Code", "Fixed_Code", "Feedback", "Constraints"
            ])
        else:
            df = pd.DataFrame(self.results)
            cols = [
                "Title", "Severity", "Confidence", "Category", 
                "File", "Line", "Description", "Suggestion", 
                "Code", "Fixed_Code", "Feedback", "Constraints"
            ]
            for c in cols:
                if c not in df.columns: df[c] = ""
            df = df[cols]
            df['Line'] = pd.to_numeric(df['Line'], errors='coerce').fillna(0)
            df.insert(0, 'S.No', range(1, 1 + len(df)))

        try:
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            workbook = writer.book

            # Summary Sheet
            summary_ws = workbook.add_worksheet('Summary')
            bold_fmt = workbook.add_format({'bold': True, 'font_size': 12})
            title_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#4F81BD'})
            date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})

            summary_ws.write('B2', 'Codebase Analysis Report', title_fmt)
            summary_ws.write('B4', 'Run ID:', bold_fmt); summary_ws.write('C4', self.run_id)
            summary_ws.write('B5', 'Date:', bold_fmt); summary_ws.write('C5', self.start_time, date_fmt)
            summary_ws.write('B6', 'Path:', bold_fmt); summary_ws.write('C6', str(self.codebase_path))
            summary_ws.write('B8', 'Issues:', bold_fmt); summary_ws.write('C8', len(self.results))
            
            # Analysis Sheet
            df.to_excel(writer, sheet_name='Analysis', index=False, startrow=1, header=False)
            worksheet = writer.sheets['Analysis']
            
            header_fmt = workbook.add_format({
                'bold': True, 'fg_color': '#4F81BD', 'font_color': '#FFFFFF', 'border': 1, 'valign': 'top'
            })
            wrap_fmt = workbook.add_format({'text_wrap': True, 'valign': 'top'})
            code_fmt = workbook.add_format({
                'font_name': 'Consolas', 'font_size': 9, 'bg_color': '#F2F2F2', 
                'text_wrap': True, 'valign': 'top', 'border': 1
            })
            
            # Headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)

            # Columns sizing
            worksheet.set_column('A:A', 6, wrap_fmt)    # S.No
            worksheet.set_column('B:B', 25, wrap_fmt)   # Title
            worksheet.set_column('C:E', 15, wrap_fmt)   # Sev/Conf/Cat
            worksheet.set_column('F:F', 20, wrap_fmt)   # File
            worksheet.set_column('G:G', 8, wrap_fmt)    # Line
            worksheet.set_column('H:I', 40, wrap_fmt)   # Desc/Sugg
            worksheet.set_column('J:K', 50, code_fmt)   # Code
            worksheet.set_column('L:M', 35, wrap_fmt)   # Feedback

            # Conditional Formatting
            max_row = len(df) + 1
            rng = f"C2:C{max_row}"
            worksheet.conditional_format(rng, {'type': 'text', 'criteria': 'containing', 'value': 'Critical', 'format': workbook.add_format({'bg_color': '#FFC7CE'})})
            worksheet.conditional_format(rng, {'type': 'text', 'criteria': 'containing', 'value': 'High', 'format': workbook.add_format({'bg_color': '#FFC7CE'})})
            
            worksheet.autofilter(0, 0, max_row, len(df.columns) - 1)
            writer.close()
            print(f"[*] Success! Report saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating Excel: {e}")
            df.to_csv(output_path.replace(".xlsx", ".csv"), index=False)

        return output_path

    def _trigger_email_report(self, recipient: str, attachment_path: str, file_count: int):
        """
        Sends a comprehensive HTML email report with detailed stats and actionable next steps.
        Matches the visual style of the Fixer Agent report.
        """
        try:
            # --- 1. Statistics Calculation ---
            total_issues = len(self.results)
            # Safe severity check (handles mixed case strings)
            crit_count = sum(1 for i in self.results if "critical" in str(i.get("Severity", "")).lower())
            high_count = sum(1 for i in self.results if "high" in str(i.get("Severity", "")).lower())
            
            # Map stats to the visual dashboard cards
            stats = {
                "Files Scanned": file_count,
                "Total Issues Detected": total_issues,
                "Critical Severity": crit_count,
                "High Severity": high_count
            }
            
            # --- 2. Enhanced Metadata (The "Run Details" Table) ---
            end_time = datetime.now()
            duration_str = str(end_time - self.start_time).split('.')[0]
            
            metadata = {
                "Agent Type": "CodebaseLLMAgent (Analysis Only)",
                "Execution Mode": "LIVE (Read-Only Scan)",
                "Project Root": self.codebase_path.name,
                "Full Path": str(self.codebase_path),
                "Start Time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "End Time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Duration": duration_str,
                "Report File": os.path.basename(attachment_path)
            }
            
            # --- 3. Comprehensive Summary / Post-Processing (The Text Block) ---
            if total_issues == 0:
                summary_intro = "✅ <b>Clean Scan:</b> No issues detected based on current heuristics."
            else:
                summary_intro = (
                    f"The agent successfully scanned {file_count} files and identified <b>{total_issues} potential issues</b>.<br>"
                    f"Top concerns include <b>{crit_count} Critical</b> and <b>{high_count} High</b> severity items."
                )

            summary = (
                f"<b>Execution Summary:</b><br>"
                f"{summary_intro}<br><br>"
                
                f"<b>Post-Processing Action Items:</b><br>"
                f"1. <b>Triage:</b> Open the attached <i>{os.path.basename(attachment_path)}</i>. Use Excel filters to isolate 'Critical' and 'High' severity rows.<br>"
                f"2. <b>Validation:</b> Verify the 'Code' and 'Suggestion' columns to confirm validity (check for false positives).<br>"
                f"3. <b>Assignment:</b> Create tickets for valid Critical issues. The 'Fixed_Code' column can be used as a starting point for developers.<br>"
                f"4. <b>Remediation:</b> For automated fixing, feed the verified rows into the <i>CodebaseFixerAgent</i> pipeline."
            )

            # --- 4. Send Email ---
            reporter = CodebaseEmailReporter()
            success = reporter.send_report(
                recipients=recipient,
                attachment_path=attachment_path,
                metadata=metadata,
                stats=stats,
                analysis_summary=summary
            )
            
            if success:
                print(f"    [Report] Success")
            else:
                print("    [!] Report Failed.")

        except Exception as e:
            print(f"[!] Error during report generation: {e}")