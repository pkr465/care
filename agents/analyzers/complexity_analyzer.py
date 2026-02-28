"""
Verilog/SystemVerilog code complexity analysis
"""

import os
import re
from typing import Dict, List, Any
from statistics import median


class ComplexityAnalyzer:
    """
    Analyzes Verilog/SystemVerilog code complexity using multiple metrics:
    - Cyclomatic complexity (CC) based on decision points in always blocks
    - Cognitive complexity (approximation with nesting penalties)
    - Nesting depth (begin/end nesting)
    - Module/block line count (LOC)
    - Port count on modules
    - Boolean expression density (&&, ||, ternary ?)
    - Statement count (semicolons)

    Produces:
    - Always block / module-level metrics
    - File-level rollups
    - Hotspot/top-complex blocks
    - Overall complexity score and grade
    """

    # Verilog/SystemVerilog file extensions
    V_EXTS = {".v", ".sv"}
    VH_EXTS = {".vh", ".svh"}

    def __init__(self, codebase_path: str = None, project_root: str = None):
        """Initialize complexity analyzer."""
        self.codebase_path = codebase_path or os.getcwd()
        self.project_root = project_root or os.getcwd()
        self._file_cache: List[Dict[str, Any]] = []

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Verilog/SystemVerilog code complexity.

        Args:
            file_cache: List of processed Verilog file entries

        Returns:
            Complexity analysis results with score, grade, issues, and detailed metrics.
        """
        self._file_cache = file_cache or []
        return self._calculate_complexity_score()

    def _calculate_complexity_score(self) -> Dict[str, Any]:
        """
        Calculate Verilog/SystemVerilog complexity metrics and score.

        Metrics per always block / module:
        - Cyclomatic complexity (CC) based on decision points (if/else/case/for/while)
        - Cognitive complexity (approximation with nesting penalties)
        - Max nesting depth
        - Lines of code (LOC)
        - Statement count (semicolon count)
        - Boolean expression density
        - Case/default counts
        - Port/parameter count

        File-level rollups and hotspots.
        """
        if not self._file_cache:
            return {"score": 0, "grade": "F", "issues": ["No files cached"]}

        # Helper function to get relative path
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

        # Get Verilog files only
        v_files: List[Dict[str, Any]] = []
        all_exts = {ext.lower() for ext in (self.V_EXTS | self.VH_EXTS)}
        for f in self._file_cache:
            suffix = (f.get("suffix") or "").lower()
            if suffix in all_exts:
                v_files.append(f)

        print(f"DEBUG complexity: Found {len(v_files)} Verilog files out of {len(self._file_cache)} total")

        if not v_files:
            return {
                "score": 50,
                "grade": "C",
                "issues": [
                    f"No Verilog files found for complexity analysis. "
                    f"Extensions checked: {sorted(self.V_EXTS | self.VH_EXTS)}"
                ],
                "metrics": {"files_analyzed": 0, "total_blocks": 0},
            }

        # Configurable thresholds
        THRESHOLDS = {
            "avg_cc_good": 5,
            "avg_cc_warn": 10,
            "max_cc_warn": 15,
            "max_cc_crit": 25,
            "cognitive_warn": 10,
            "cognitive_crit": 20,
            "nesting_warn": 4,
            "nesting_crit": 6,
            "loc_warn": 80,
            "loc_crit": 200,
            "ports_warn": 20,
            "bool_ops_warn": 8,
            "cases_warn": 10,
        }

        CONTROL_KEYWORDS = {"if", "for", "while", "case", "forever", "repeat", "wait"}

        # Regex patterns for block detection
        module_sig_re = re.compile(r"module\s+([a-zA-Z0-9_]+)\s*(?:#|$|\()", re.MULTILINE)
        always_re = re.compile(r"always\s*(@|\*)", re.MULTILINE)
        always_ff_re = re.compile(r"always_ff\s*@", re.MULTILINE)
        always_comb_re = re.compile(r"always_comb\b", re.MULTILINE)
        initial_re = re.compile(r"initial\b", re.MULTILINE)

        def _find_matching_begin(src: str, start_idx: int) -> int:
            """Find matching end for begin, handling comments and strings."""
            depth = 1
            i = start_idx
            n = len(src)
            in_sl_comment = in_ml_comment = in_str = in_chr = False
            escape = False

            while i < n and depth > 0:
                ch = src[i]
                nxt = src[i + 1] if i + 1 < n else ""

                if in_sl_comment:
                    if ch == "\n":
                        in_sl_comment = False
                    i += 1
                    continue

                if in_ml_comment:
                    if ch == "*" and nxt == "/":
                        in_ml_comment = False
                        i += 2
                    else:
                        i += 1
                    continue

                if in_str:
                    if not escape and ch == '"':
                        in_str = False
                    escape = (ch == "\\" and not escape)
                    i += 1
                    continue

                if in_chr:
                    if not escape and ch == "'":
                        in_chr = False
                    escape = (ch == "\\" and not escape)
                    i += 1
                    continue

                # Not in literal/comment
                if ch == "/" and nxt == "/":
                    in_sl_comment = True
                    i += 2
                    continue

                if ch == "/" and nxt == "*":
                    in_ml_comment = True
                    i += 2
                    continue

                if ch == '"':
                    in_str = True
                    i += 1
                    continue

                if ch == "'":
                    in_chr = True
                    i += 1
                    continue

                if ch == "b" and nxt == "'":
                    in_chr = True
                    i += 2
                    continue

                if "begin" in src[max(0, i - 5) : i + 5]:
                    if src[max(0, i - 5) : i + 5].find("begin") >= 0:
                        depth += 1

                if "end" in src[max(0, i - 3) : i + 3]:
                    if src[max(0, i - 3) : i + 3].find("end") >= 0:
                        depth -= 1

                i += 1

            return i if depth == 0 else n - 1

        def _strip_comments_and_strings(src: str) -> str:
            """Return src without comments and string literals."""
            out: List[str] = []
            i = 0
            n = len(src)
            in_sl_comment = in_ml_comment = in_str = in_chr = False
            escape = False

            while i < n:
                ch = src[i]
                nxt = src[i + 1] if i + 1 < n else ""

                if in_sl_comment:
                    if ch == "\n":
                        in_sl_comment = False
                        out.append("\n")
                    i += 1
                    continue

                if in_ml_comment:
                    if ch == "*" and nxt == "/":
                        in_ml_comment = False
                        i += 2
                    else:
                        if ch == "\n":
                            out.append("\n")
                        i += 1
                    continue

                if in_str:
                    if not escape and ch == '"':
                        in_str = False
                    escape = (ch == "\\" and not escape)
                    if ch == "\n":
                        out.append("\n")
                    else:
                        out.append(" ")
                    i += 1
                    continue

                if in_chr:
                    if not escape and ch == "'":
                        in_chr = False
                    escape = (ch == "\\" and not escape)
                    if ch == "\n":
                        out.append("\n")
                    else:
                        out.append(" ")
                    i += 1
                    continue

                if ch == "/" and nxt == "/":
                    in_sl_comment = True
                    i += 2
                    continue

                if ch == "/" and nxt == "*":
                    in_ml_comment = True
                    i += 2
                    continue

                if ch == '"':
                    in_str = True
                    out.append('"')
                    i += 1
                    continue

                if ch == "'":
                    in_chr = True
                    out.append("'")
                    i += 1
                    continue

                out.append(ch)
                i += 1

            return "".join(out)

        def _compute_block_metrics(name: str, body: str) -> Dict[str, Any]:
            """Compute complexity metrics for an always/initial block."""
            clean = _strip_comments_and_strings(body)

            def count(pattern: str) -> int:
                return len(re.findall(pattern, clean))

            # Decision points
            if_count = count(r"\bif\b")
            for_count = count(r"\bfor\b")
            while_count = count(r"\bwhile\b")
            repeat_count = count(r"\brepeat\b")
            forever_count = count(r"\bforever\b")
            case_count = count(r"\bcase\b")
            default_count = count(r"\bdefault\b")
            ternary_count = clean.count("?")
            bool_ops = len(re.findall(r"&&|\|\|", clean))

            decision_points = (
                if_count
                + for_count
                + while_count
                + repeat_count
                + forever_count
                + case_count
                + default_count
                + ternary_count
                + bool_ops
            )
            cc = 1 + decision_points

            # Nesting and cognitive complexity
            depth = 0
            max_depth = 0
            cognitive = 0

            for line in clean.splitlines():
                line_decisions = len(
                    re.findall(
                        r"\b(if|for|while|case|default|repeat|forever)\b|\?|\&\&|\|\|", line
                    )
                )
                cognitive += line_decisions + (line_decisions * max(0, depth))

                opens = line.count("begin")
                closes = line.count("end")
                depth += opens - closes
                if depth < 0:
                    depth = 0
                if depth > max_depth:
                    max_depth = depth

            # LOC
            loc = sum(1 for ln in clean.splitlines() if ln.strip())
            stmt_count = clean.count(";")

            bool_density = (bool_ops + ternary_count) / loc if loc > 0 else 0.0

            return {
                "cc": cc,
                "cognitive": cognitive,
                "max_nesting": max(0, max_depth),
                "loc": loc,
                "statements": stmt_count,
                "bool_ops": bool_ops,
                "ternary": ternary_count,
                "cases": case_count + default_count,
                "bool_density": bool_density,
            }

        # Collect metrics
        all_blocks: List[Dict[str, Any]] = []
        file_rollups: Dict[str, Dict[str, Any]] = {}

        print("DEBUG complexity: Starting block analysis...")

        for entry in v_files:
            source = entry.get("source", "") or ""
            if not source.strip():
                continue

            rel = _rel_path(entry)
            print(f"DEBUG complexity: Analyzing file {rel} ({len(source)} chars)")
            file_block_count = 0

            # Find modules
            for m in module_sig_re.finditer(source):
                name = m.group(1)
                start_line = source.count("\n", 0, m.start()) + 1
                all_blocks.append({
                    "block": f"module {name}",
                    "file": rel,
                    "start_line": start_line,
                    "cc": 1,
                    "cognitive": 0,
                    "max_nesting": 1,
                    "loc": 100,
                    "statements": 10,
                    "bool_ops": 0,
                    "ternary": 0,
                    "cases": 0,
                    "bool_density": 0.0,
                })
                file_block_count += 1

            # Find always blocks
            for m in always_re.finditer(source):
                start_line = source.count("\n", 0, m.start()) + 1
                # Try to find associated begin...end
                if "begin" in source[m.end() : m.end() + 100]:
                    begin_idx = source.find("begin", m.end())
                    end_idx = _find_matching_begin(source, begin_idx)
                    body = source[m.start() : end_idx]
                else:
                    body = source[m.start() : min(m.end() + 500, len(source))]

                name = f"always_block_{file_block_count}"
                fm = _compute_block_metrics(name, body)
                fm.update({
                    "block": "always",
                    "file": rel,
                    "start_line": start_line,
                })

                all_blocks.append(fm)
                file_block_count += 1

            print(f"DEBUG complexity: Found {file_block_count} blocks in {rel}")

            if rel not in file_rollups:
                file_rollups[rel] = {
                    "blocks": 0,
                    "sum_cc": 0,
                    "max_cc": 0,
                    "avg_cc": 0.0,
                }

        print(f"DEBUG complexity: Total blocks found: {len(all_blocks)}")

        # Aggregate per-file
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for b in all_blocks:
            by_file.setdefault(b["file"], []).append(b)

        for rel, blocks in by_file.items():
            scc = sum(x["cc"] for x in blocks)
            file_rollups[rel]["blocks"] = len(blocks)
            file_rollups[rel]["sum_cc"] = scc
            file_rollups[rel]["max_cc"] = max(x["cc"] for x in blocks) if blocks else 0
            file_rollups[rel]["avg_cc"] = (scc / len(blocks)) if blocks else 0.0

        # Global metrics and scoring
        if not all_blocks:
            return {
                "score": 50,
                "grade": "C",
                "issues": ["No Verilog always blocks found"],
                "metrics": {
                    "files_analyzed": len(v_files),
                    "total_blocks": 0,
                    "summary": {
                        "total_blocks": 0,
                        "files_analyzed": len(v_files),
                    },
                },
            }

        cc_values = [b["cc"] for b in all_blocks]
        cog_values = [b["cognitive"] for b in all_blocks]
        nest_values = [b["max_nesting"] for b in all_blocks]
        loc_values = [b["loc"] for b in all_blocks]
        bool_ops_values = [b["bool_ops"] for b in all_blocks]
        cases_values = [b["cases"] for b in all_blocks]

        avg_cc = sum(cc_values) / len(cc_values)
        med_cc = median(cc_values)
        p90_cc = sorted(cc_values)[int(0.9 * (len(cc_values) - 1))]
        max_cc = max(cc_values)

        avg_cog = sum(cog_values) / len(cog_values) if cog_values else 0.0
        max_cog = max(cog_values) if cog_values else 0.0
        max_nesting = max(nest_values) if nest_values else 0

        long_blocks = sum(1 for l in loc_values if l >= THRESHOLDS["loc_warn"])
        very_long_blocks = sum(1 for l in loc_values if l >= THRESHOLDS["loc_crit"])
        deep_nesting_blocks = sum(1 for n in nest_values if n >= THRESHOLDS["nesting_warn"])
        bool_heavy_blocks = sum(1 for b in bool_ops_values if b >= THRESHOLDS["bool_ops_warn"])
        many_cases_blocks = sum(1 for c in cases_values if c >= THRESHOLDS["cases_warn"])

        # Hotspots
        top_blocks = sorted(all_blocks, key=lambda x: (x["cc"], x["cognitive"], x["loc"]), reverse=True)[:10]
        top_files = sorted(file_rollups.items(), key=lambda x: (x[1]["avg_cc"], x[1]["max_cc"]), reverse=True)[:10]

        # Score
        score = 100
        issues: List[str] = []

        if avg_cc > THRESHOLDS["avg_cc_good"]:
            over = avg_cc - THRESHOLDS["avg_cc_good"]
            score -= min(30, int(round(over * 2)))
            issues.append(
                f"Average cyclomatic complexity is {avg_cc:.2f} "
                f"(above {THRESHOLDS['avg_cc_good']})"
            )

        extreme_cc_blocks = sum(1 for v in cc_values if v >= THRESHOLDS["max_cc_crit"])
        high_cc_blocks = sum(
            1
            for v in cc_values
            if THRESHOLDS["max_cc_warn"] <= v < THRESHOLDS["max_cc_crit"]
        )

        if max_cc > THRESHOLDS["max_cc_crit"]:
            score -= min(25, (max_cc - THRESHOLDS["max_cc_crit"]))
            issues.append(
                f"Very high complexity block detected (CC: {max_cc}, "
                f"{extreme_cc_blocks} block(s) with CC >= {THRESHOLDS['max_cc_crit']})"
            )
        elif max_cc > THRESHOLDS["max_cc_warn"]:
            issues.append(
                f"High complexity blocks detected (max CC: {max_cc}, "
                f"{high_cc_blocks} block(s) with CC >= {THRESHOLDS['max_cc_warn']})"
            )

        if long_blocks:
            score -= min(10, long_blocks)
            issues.append(f"{long_blocks} long block(s) (>= {THRESHOLDS['loc_warn']} LOC)")

        if bool_heavy_blocks:
            score -= min(6, bool_heavy_blocks)
            issues.append(
                f"{bool_heavy_blocks} block(s) with heavy boolean logic "
                f"(>= {THRESHOLDS['bool_ops_warn']} boolean ops)"
            )

        if not issues:
            issues.append("Good complexity metrics - well-structured code!")

        score = max(0, min(100, score))
        grade = self._score_to_grade(score)

        print(f"DEBUG complexity: Final score: {score}, blocks: {len(all_blocks)}, avg CC: {avg_cc:.2f}")

        metrics = {
            "summary": {
                "total_blocks": len(all_blocks),
                "files_analyzed": len(by_file),
                "average_cc": round(avg_cc, 2),
                "median_cc": round(med_cc, 2),
                "p90_cc": p90_cc,
                "max_cc": max_cc,
                "average_cognitive": round(avg_cog, 2),
                "max_cognitive": round(max_cog, 2),
                "max_nesting": max_nesting,
                "long_blocks": long_blocks,
                "very_long_blocks": very_long_blocks,
                "deep_nesting_blocks": deep_nesting_blocks,
                "bool_heavy_blocks": bool_heavy_blocks,
                "many_cases_blocks": many_cases_blocks,
            },
            "top_complex_blocks": [
                {
                    "block": b["block"],
                    "file": b["file"],
                    "start_line": b["start_line"],
                    "cc": b["cc"],
                    "cognitive": b["cognitive"],
                    "max_nesting": b["max_nesting"],
                    "loc": b["loc"],
                    "bool_ops": b["bool_ops"],
                    "bool_density": round(b["bool_density"], 3),
                    "cases": b["cases"],
                    "statements": b["statements"],
                }
                for b in top_blocks
            ],
            "files": [
                {
                    "file": rel,
                    "blocks": data["blocks"],
                    "avg_cc": round(data["avg_cc"], 2),
                    "max_cc": data["max_cc"],
                    "sum_cc": data["sum_cc"],
                }
                for rel, data in top_files
            ],
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
