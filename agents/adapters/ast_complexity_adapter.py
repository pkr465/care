"""HDL complexity analysis using Verilator (with regex fallback)."""

import logging
import re
import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from agents.adapters.base_adapter import BaseStaticAdapter

# Check for Verilator availability
import shutil
VERILATOR_PATH = shutil.which("verilator")
VERILATOR_AVAILABLE = VERILATOR_PATH is not None


class HDLComplexityAdapter(BaseStaticAdapter):
    """
    Analyzes Verilog/SystemVerilog code complexity using Verilator (with regex fallback).

    Measures cyclomatic complexity in always blocks, nesting depth, port count, module length.
    Flags high-complexity always blocks and deep nesting as red flags.

    Falls back to regex-based analysis when Verilator is not installed.
    """

    # Module definition regex
    _MODULE_RE = re.compile(
        r'module\s+(\w+)\s*(?:#\s*\([\s\S]*?\))?\s*\(',
        re.MULTILINE
    )

    # Always block detection (sequential, combinational, latches)
    _ALWAYS_RE = re.compile(r'always\s*@\s*\(', re.MULTILINE)
    _ALWAYS_FF_RE = re.compile(r'always_ff\s*@', re.MULTILINE)
    _ALWAYS_COMB_RE = re.compile(r'always_comb\s*\(', re.MULTILINE)
    _ALWAYS_LATCH_RE = re.compile(r'always_latch\s*\(', re.MULTILINE)

    # Task/function detection
    _TASK_RE = re.compile(r'task\s+(\w+)', re.MULTILINE)
    _FUNCTION_RE = re.compile(r'function\s+.*?(\w+)\s*\(', re.MULTILINE)

    # Decision-point keywords that contribute to cyclomatic complexity in always blocks
    _CC_KEYWORDS = re.compile(
        r'\b(?:if|else\s+if|for|while|case|begin|end|\?\s*:)\b'
    )

    def __init__(self, debug: bool = False):
        """
        Initialize HDL complexity adapter.

        Args:
            debug: Enable debug logging if True.
        """
        super().__init__("hdl_complexity", debug=debug)
        self.verilator_available = VERILATOR_AVAILABLE
        if not self.verilator_available:
            self.logger.warning(
                "Verilator not available — using regex fallback. "
                "For best results: install Verilator"
            )

    # ── Regex-based fallback ──────────────────────────────────────────────

    def _find_matching_brace(self, source: str, open_pos: int) -> int:
        """Find the closing brace matching the one at *open_pos*."""
        depth = 0
        in_string = False
        in_char = False
        in_line_comment = False
        in_block_comment = False
        i = open_pos
        while i < len(source):
            c = source[i]
            # Handle string/char/comment state
            if in_line_comment:
                if c == '\n':
                    in_line_comment = False
            elif in_block_comment:
                if c == '*' and i + 1 < len(source) and source[i + 1] == '/':
                    in_block_comment = False
                    i += 1
            elif in_string:
                if c == '\\':
                    i += 1  # skip escaped char
                elif c == '"':
                    in_string = False
            elif in_char:
                if c == '\\':
                    i += 1
                elif c == "'":
                    in_char = False
            else:
                if c == '/' and i + 1 < len(source):
                    nxt = source[i + 1]
                    if nxt == '/':
                        in_line_comment = True
                    elif nxt == '*':
                        in_block_comment = True
                elif c == '"':
                    in_string = True
                elif c == "'":
                    in_char = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    def _compute_max_nesting(self, body: str) -> int:
        """Estimate max nesting depth from brace counts in function body."""
        max_depth = 0
        depth = 0
        in_string = False
        in_line_comment = False
        in_block_comment = False
        i = 0
        while i < len(body):
            c = body[i]
            if in_line_comment:
                if c == '\n':
                    in_line_comment = False
            elif in_block_comment:
                if c == '*' and i + 1 < len(body) and body[i + 1] == '/':
                    in_block_comment = False
                    i += 1
            elif in_string:
                if c == '\\':
                    i += 1
                elif c == '"':
                    in_string = False
            else:
                if c == '/' and i + 1 < len(body):
                    nxt = body[i + 1]
                    if nxt == '/':
                        in_line_comment = True
                    elif nxt == '*':
                        in_block_comment = True
                elif c == '"':
                    in_string = True
                elif c == '{':
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif c == '}':
                    depth -= 1
            i += 1
        return max_depth

    def _regex_analyze_file(self, file_path: str, source: str) -> List[Dict]:
        """Extract HDL metrics using regex (fallback when Verilator unavailable)."""
        blocks = []

        # Extract module definitions
        for m in self._MODULE_RE.finditer(source):
            module_name = m.group(1)
            start_line = source[:m.start()].count('\n') + 1

            # Find matching 'endmodule'
            endmodule_pos = source.find('endmodule', m.start())
            if endmodule_pos == -1:
                continue

            end_line = source[:endmodule_pos].count('\n') + 1
            module_body = source[m.start():endmodule_pos]
            length = end_line - start_line + 1

            # Count ports (parameters in module definition)
            port_match = re.search(r'\(([\s\S]*?)\)', source[m.start():m.start() + 200])
            port_count = 0
            if port_match:
                port_str = port_match.group(1)
                port_count = port_str.count(',') + (1 if port_str.strip() else 0)

            # Count always blocks in this module
            always_count = len(self._ALWAYS_RE.findall(module_body))
            always_ff_count = len(self._ALWAYS_FF_RE.findall(module_body))
            always_comb_count = len(self._ALWAYS_COMB_RE.findall(module_body))

            # Cyclomatic complexity: count decision points in always blocks
            cc = 1 + len(self._CC_KEYWORDS.findall(module_body))

            # Max nesting depth
            nesting = self._compute_max_nesting(module_body)

            blocks.append({
                "file": file_path,
                "name": module_name,
                "long_name": module_name,
                "start_line": start_line,
                "end_line": end_line,
                "length": length,
                "cyclomatic_complexity": cc,
                "token_count": len(module_body.split()),
                "parameter_count": port_count,
                "max_nesting_depth": nesting,
                "always_blocks": always_count,
                "always_ff_blocks": always_ff_count,
                "always_comb_blocks": always_comb_count,
            })

        return blocks

    # ── Main entry point ──────────────────────────────────────────────────

    def analyze(
        self,
        file_cache: List[Dict[str, Any]],
        verible_parser: Optional[Any] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog files for complexity metrics.

        Uses Verilator when available; falls back to regex-based analysis otherwise.

        Args:
            file_cache: List of file entries with "file_relative_path" and "source" keys.
            verible_parser: Optional Verible parser (unused here).
            dependency_graph: Optional dependency graph (unused here).

        Returns:
            Standard analysis result dict with score, grade, metrics, issues, details.
        """

        using_fallback = not self.verilator_available

        # Filter to Verilog/SystemVerilog files
        verilog_suffixes = (".v", ".sv", ".svh", ".vh")
        verilog_files = [
            entry
            for entry in file_cache
            if entry.get("file_relative_path", "").endswith(verilog_suffixes)
        ]

        if not verilog_files:
            return self._empty_result("No Verilog/SystemVerilog files to analyze")

        # Analyze each file and collect module metrics
        all_modules = []
        for entry in verilog_files:
            file_path = entry.get("file_relative_path", "unknown")
            source_code = entry.get("source", "")

            try:
                if self.verilator_available:
                    # Try Verilator analysis
                    modules = self._verilator_analyze_file(file_path, source_code)
                    if modules:
                        all_modules.extend(modules)
                    else:
                        # Fallback to regex if Verilator produces no output
                        modules = self._regex_analyze_file(file_path, source_code)
                        all_modules.extend(modules)
                else:
                    # Regex fallback
                    modules = self._regex_analyze_file(file_path, source_code)
                    all_modules.extend(modules)

                if self.debug:
                    self.logger.debug(f"Analyzed {file_path}: modules found so far {len(all_modules)}")

            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
                continue

        if not all_modules:
            return self._empty_result("No modules found in Verilog/SystemVerilog files")

        # Identify flagged modules
        flagged_modules = []
        details = []
        high_cc_count = 0
        critical_cc_count = 0
        deep_nesting_count = 0
        many_ports_count = 0
        large_module_count = 0

        for module in all_modules:
            cc = module["cyclomatic_complexity"]
            nesting = module["max_nesting_depth"]
            ports = module["parameter_count"]
            loc = module["length"]

            flagged = False
            issue_strs = []

            # Flag high cyclomatic complexity (adjusted thresholds for always blocks)
            if cc > 20:
                critical_cc_count += 1
                flagged = True
                issue_strs.append(f"critical complexity ({cc})")
            elif cc > 10:
                high_cc_count += 1
                flagged = True
                issue_strs.append(f"high complexity ({cc})")

            # Flag deep nesting
            if nesting > 4:
                deep_nesting_count += 1
                flagged = True
                issue_strs.append(f"deep nesting (depth {nesting})")

            # Flag many ports
            if ports > 50:
                many_ports_count += 1
                flagged = True
                issue_strs.append(f"many ports ({ports})")

            # Flag long modules
            if loc > 300:
                large_module_count += 1
                flagged = True
                issue_strs.append(f"large module ({loc} LOC)")

            if flagged:
                description = f"{module['long_name']}: {', '.join(issue_strs)}"
                detail = self._make_detail(
                    file=module["file"],
                    module=module["long_name"],
                    line=module["start_line"],
                    description=description,
                    severity=self._severity_from_cc(cc),
                    category="complexity",
                    drc="",
                )
                details.append(detail)
                flagged_modules.append(module)

        # Calculate score
        score = 100.0
        score -= high_cc_count * 5
        score -= critical_cc_count * 10
        score -= deep_nesting_count * 2
        score -= many_ports_count * 2
        score -= large_module_count * 3
        score = max(0, min(100, score))

        # Compute metrics
        avg_cc = sum(m["cyclomatic_complexity"] for m in all_modules) / len(all_modules)
        max_cc = max(m["cyclomatic_complexity"] for m in all_modules)
        cc_list = sorted([m["cyclomatic_complexity"] for m in all_modules])
        median_cc = cc_list[len(cc_list) // 2]

        avg_nesting = sum(m["max_nesting_depth"] for m in all_modules) / len(all_modules)
        avg_ports = sum(m["parameter_count"] for m in all_modules) / len(all_modules)
        avg_loc = sum(m["length"] for m in all_modules) / len(all_modules)

        total_always_blocks = sum(m.get("always_blocks", 0) for m in all_modules)
        total_always_ff = sum(m.get("always_ff_blocks", 0) for m in all_modules)
        total_always_comb = sum(m.get("always_comb_blocks", 0) for m in all_modules)

        metrics = {
            "tool_available": True,
            "analysis_mode": "regex_fallback" if using_fallback else "verilator",
            "files_analyzed": len(verilog_files),
            "modules_analyzed": len(all_modules),
            "avg_cyclomatic_complexity": round(avg_cc, 2),
            "max_cyclomatic_complexity": int(max_cc),
            "median_cyclomatic_complexity": int(median_cc),
            "high_cc_count": high_cc_count,
            "critical_cc_count": critical_cc_count,
            "deep_nesting_count": deep_nesting_count,
            "many_ports_count": many_ports_count,
            "large_module_count": large_module_count,
            "avg_nesting_depth": round(avg_nesting, 2),
            "avg_ports": round(avg_ports, 2),
            "avg_lines_of_code": round(avg_loc, 2),
            "total_always_blocks": total_always_blocks,
            "total_always_ff_blocks": total_always_ff,
            "total_always_comb_blocks": total_always_comb,
        }

        # Build issues list
        issues = []
        if critical_cc_count > 0:
            issues.append(
                f"Found {critical_cc_count} module(s) with critical complexity (CC > 20)"
            )
        if high_cc_count > 0:
            issues.append(
                f"Found {high_cc_count} module(s) with high complexity (CC > 10)"
            )
        if deep_nesting_count > 0:
            issues.append(
                f"Found {deep_nesting_count} module(s) with deep nesting (depth > 4)"
            )
        if many_ports_count > 0:
            issues.append(
                f"Found {many_ports_count} module(s) with many ports (> 50)"
            )
        if large_module_count > 0:
            issues.append(
                f"Found {large_module_count} module(s) with large size (> 300 LOC)"
            )
        if not issues:
            issues = ["All modules within acceptable complexity thresholds"]

        grade = self._score_to_grade(score)

        return {
            "score": score,
            "grade": grade,
            "metrics": metrics,
            "issues": issues,
            "details": details,
            "tool_available": True,
        }

    def _verilator_analyze_file(self, file_path: str, source_code: str) -> List[Dict]:
        """
        Use Verilator --xml-only to analyze file and extract AST metrics.
        Falls back to regex if XML parsing fails.
        """
        tmp_path = None
        xml_path = None
        try:
            # Write source to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.sv', delete=False, encoding='utf-8'
            ) as tmp:
                tmp.write(source_code)
                tmp_path = tmp.name

            # Create temp directory for Verilator output
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract module name for Verilator prefix (simplified: use 'V' + hash)
                prefix = "V" + str(abs(hash(file_path)) % 100000)

                # Run Verilator with --xml-only
                cmd = [
                    VERILATOR_PATH,
                    '--xml-only',
                    '-Wall',
                    f'--prefix={prefix}',
                    f'--Mdir={tmpdir}',
                    tmp_path
                ]

                if self.debug:
                    self.logger.debug(f"Running Verilator: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )

                if result.returncode != 0:
                    if self.debug:
                        self.logger.debug(
                            f"Verilator returned {result.returncode}: {result.stderr[:200]}"
                        )
                    # Fall through to regex fallback
                    return []

                # Look for generated XML file
                xml_files = [f for f in os.listdir(tmpdir) if f.endswith('.xml')]
                if not xml_files:
                    if self.debug:
                        self.logger.debug("No XML output from Verilator")
                    return []

                xml_path = os.path.join(tmpdir, xml_files[0])

                # Parse XML and extract metrics
                modules = self._parse_verilator_xml(xml_path, file_path)
                return modules

        except subprocess.TimeoutExpired:
            if self.debug:
                self.logger.debug(f"Verilator timeout for {file_path}")
            return []
        except Exception as e:
            if self.debug:
                self.logger.debug(f"Verilator analysis failed for {file_path}: {e}")
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _parse_verilator_xml(self, xml_path: str, file_path: str) -> List[Dict]:
        """
        Parse Verilator XML AST output and extract complexity metrics.

        Returns list of module dicts with keys:
        - file, name, long_name, start_line, end_line, length
        - cyclomatic_complexity, token_count, parameter_count, max_nesting_depth
        - always_blocks, always_ff_blocks, always_comb_blocks
        - signal_count, expression_depth, operator_count, instantiation_count (AST-derived)
        """
        modules = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Navigate to netlist
            netlist = root.find('netlist')
            if netlist is None:
                return []

            # Process each module
            for module_elem in netlist.findall('module'):
                module_name = module_elem.get('name', 'unknown')
                orig_name = module_elem.get('origName', module_name)

                # Extract module metrics from AST
                module_dict = {
                    'file': file_path,
                    'name': module_name,
                    'long_name': orig_name,
                    'start_line': 1,  # XML doesn't track source line numbers
                    'end_line': 1,
                    'length': 0,
                    'cyclomatic_complexity': 1,
                    'token_count': 0,
                    'parameter_count': 0,
                    'max_nesting_depth': 0,
                    'always_blocks': 0,
                    'always_ff_blocks': 0,
                    'always_comb_blocks': 0,
                    'signal_count': 0,
                    'expression_depth': 0,
                    'operator_count': 0,
                    'instantiation_count': 0,
                }

                # Count ports (input/output/inout var nodes)
                port_count = 0
                for var in module_elem.findall('.//var'):
                    var_dir = var.get('dir', '')
                    if var_dir in ('input', 'output', 'inout'):
                        port_count += 1
                module_dict['parameter_count'] = port_count

                # Count all signals (var elements)
                signal_count = len(module_elem.findall('.//var'))
                module_dict['signal_count'] = signal_count

                # Count always blocks by type
                always_count = len(module_elem.findall('.//always'))
                always_ff_count = len(module_elem.findall('.//alwaysff'))
                always_comb_count = len(module_elem.findall('.//alwayscomb'))
                module_dict['always_blocks'] = always_count
                module_dict['always_ff_blocks'] = always_ff_count
                module_dict['always_comb_blocks'] = always_comb_count

                # Count instantiations
                instantiation_count = len(module_elem.findall('.//instance'))
                module_dict['instantiation_count'] = instantiation_count

                # Compute cyclomatic complexity from control flow nodes
                cc = self._compute_cc_from_ast(module_elem)
                module_dict['cyclomatic_complexity'] = cc

                # Compute max nesting depth from AST structure
                max_depth = self._compute_nesting_from_ast(module_elem)
                module_dict['max_nesting_depth'] = max_depth

                # Compute expression depth
                expr_depth = self._compute_expression_depth_from_ast(module_elem)
                module_dict['expression_depth'] = expr_depth

                # Count operators in expressions
                operator_count = self._count_operators_in_ast(module_elem)
                module_dict['operator_count'] = operator_count

                # Estimate token count from AST size (rough proxy)
                token_count = len(module_elem.findall('.//*'))
                module_dict['token_count'] = token_count

                modules.append(module_dict)

            return modules

        except ET.ParseError as e:
            if self.debug:
                self.logger.debug(f"XML parse error: {e}")
            return []
        except Exception as e:
            if self.debug:
                self.logger.debug(f"Error parsing Verilator XML: {e}")
            return []

    def _compute_cc_from_ast(self, module_elem: ET.Element) -> int:
        """
        Compute cyclomatic complexity by counting control flow nodes.
        CC = 1 + number of decision points (if, case, while, for, etc.)
        """
        cc = 1

        # Count decision points in always blocks and combinational logic
        for always in module_elem.findall('.//always'):
            cc += self._count_decision_nodes(always)
        for always_ff in module_elem.findall('.//alwaysff'):
            cc += self._count_decision_nodes(always_ff)
        for always_comb in module_elem.findall('.//alwayscomb'):
            cc += self._count_decision_nodes(always_comb)

        return cc

    def _count_decision_nodes(self, elem: ET.Element) -> int:
        """Count if/case/while/for nodes in an element tree."""
        count = 0

        # Count if statements
        count += len(elem.findall('.//if'))

        # Count case statements and case items
        count += len(elem.findall('.//case'))
        count += len(elem.findall('.//caseitem'))

        # Count loops
        count += len(elem.findall('.//while'))
        count += len(elem.findall('.//for'))

        return count

    def _compute_nesting_from_ast(self, module_elem: ET.Element) -> int:
        """
        Compute max nesting depth by walking the AST tree structure.
        Nesting increases with begin/end blocks and control structures.
        """
        max_depth = 0

        def walk_depth(elem: ET.Element, depth: int = 0) -> int:
            """Recursively walk tree and track max depth."""
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            # Increment depth for structural elements
            tag = elem.tag.lower()
            next_depth = depth
            if tag in ('begin', 'if', 'else', 'for', 'while', 'case'):
                next_depth = depth + 1

            for child in elem:
                walk_depth(child, next_depth)

        for always in module_elem.findall('.//always'):
            walk_depth(always, 0)
        for always_ff in module_elem.findall('.//alwaysff'):
            walk_depth(always_ff, 0)
        for always_comb in module_elem.findall('.//alwayscomb'):
            walk_depth(always_comb, 0)

        return max_depth

    def _compute_expression_depth_from_ast(self, module_elem: ET.Element) -> int:
        """
        Compute expression depth (max nesting of operators in expressions).
        """
        max_expr_depth = 0

        def calc_expr_depth(elem: ET.Element, depth: int = 0) -> int:
            """Recursively compute expression nesting."""
            nonlocal max_expr_depth
            max_expr_depth = max(max_expr_depth, depth)

            tag = elem.tag.lower()
            # Operators and expressions increase depth
            if tag in ('add', 'sub', 'mult', 'div', 'and', 'or', 'xor', 'not',
                      'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'concat', 'replicate'):
                next_depth = depth + 1
            else:
                next_depth = depth

            for child in elem:
                calc_expr_depth(child, next_depth)

            return max_expr_depth

        for assign in module_elem.findall('.//assign'):
            calc_expr_depth(assign, 0)

        return max_expr_depth

    def _count_operators_in_ast(self, module_elem: ET.Element) -> int:
        """Count operator nodes in the AST."""
        operators = {
            'add', 'sub', 'mult', 'div', 'mod', 'pow',
            'and', 'or', 'xor', 'not', 'nand', 'nor', 'xnor',
            'eq', 'ne', 'lt', 'le', 'gt', 'ge',
            'lshift', 'rshift', 'concat', 'replicate',
            'ternary', 'reduce_and', 'reduce_or', 'reduce_xor'
        }

        count = 0
        for elem in module_elem.iter():
            if elem.tag.lower() in operators:
                count += 1

        return count

    def generate_report_data(self) -> Dict[str, Any]:
        """
        Generate structured report data suitable for PDF export.
        This is called by reporting infrastructure to produce detailed per-module breakdown.
        """
        return {
            'title': 'HDL Complexity Analysis Report',
            'tool': 'Verilator AST Analysis (with regex fallback)',
            'date': self._get_current_date(),
            'modules': [],  # Populated by caller with module data
            'summary': {
                'total_modules': 0,
                'high_complexity_modules': 0,
                'avg_cc': 0,
                'recommendation': 'Refactor high-complexity modules into smaller, focused units'
            }
        }

    @staticmethod
    def _severity_from_cc(cc: int) -> str:
        """Map cyclomatic complexity to severity."""
        if cc > 20:
            return "critical"
        elif cc > 10:
            return "high"
        elif cc > 5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _get_current_date() -> str:
        """Return current date as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
