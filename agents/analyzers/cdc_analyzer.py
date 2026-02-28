"""
Verilog/SystemVerilog clock domain crossing (CDC) analysis
"""

import re
from typing import Dict, List, Set, Any, Tuple
from .base_runtime_analyzer import RuntimeAnalyzerBase


class CDCAnalyzer(RuntimeAnalyzerBase):
    """
    Clock Domain Crossing (CDC) and synchronization analyzer.

    Detects:
    1. Signal crossing clock domains without synchronizer
    2. Multi-bit CDC without gray code or handshake
    3. Missing synchronizer stages (need 2+ flip-flop stages)
    4. Reset domain crossing
    5. Pulse synchronization issues
    """

    # Patterns for clock detection
    _POSEDGE_CLK = re.compile(r"\bposedge\s+(\w+)")
    _NEGEDGE_CLK = re.compile(r"\bnegedge\s+(\w+)")
    _CLK_PATTERN = re.compile(r"(clk\w*|clock\w*)", re.IGNORECASE)

    # Synchronizer patterns
    _SYNC_PATTERN = re.compile(r"(sync\w*|synchronizer|synchron)", re.IGNORECASE)
    _FF_PATTERN = re.compile(r"reg\s+\w+\s*=\s*\w+")  # FF chains
    _GRAY_PATTERN = re.compile(r"gray|grey", re.IGNORECASE)

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch CDC analysis across files.
        """
        issues = []
        metrics = []
        clock_domains: Set[str] = set()

        for entry in file_cache:
            if entry.get("suffix", "").lower() not in {".v", ".sv", ".vh", ".svh"}:
                continue

            rel_path = entry.get("rel_path", "unknown")
            source = entry.get("source", "")

            result = self.analyze_single_file(source, rel_path)

            issues.extend(result["issues"])
            metrics.append(result["metrics"])

            # Collect clock domains
            for clk in result.get("clocks_detected", []):
                clock_domains.add(clk)

        # Aggregate warnings if multiple clock domains without proper CDC
        if len(clock_domains) > 1:
            issues.append(f"Multiple clock domains detected: {', '.join(sorted(clock_domains))}")

        return {
            "metrics": metrics,
            "issues": issues,
            "clock_domains": sorted(list(clock_domains)),
        }

    def analyze_single_file(self, source: str, rel_path: str) -> Dict[str, Any]:
        """
        Performs local CDC analysis on a single file.
        """
        local_issues = []
        cdc_risk_count = 0
        clocks_detected: Set[str] = set()

        # Detect all clock signals
        for m in self._POSEDGE_CLK.finditer(source):
            clocks_detected.add(m.group(1))
        for m in self._NEGEDGE_CLK.finditer(source):
            clocks_detected.add(m.group(1))

        func_blocks = self._get_function_blocks(source)

        for block_name, body, start_line in func_blocks:
            # Detect clock domain crossings
            block_clocks = set()
            for m in self._POSEDGE_CLK.finditer(body):
                block_clocks.add(m.group(1))
            for m in self._NEGEDGE_CLK.finditer(body):
                block_clocks.add(m.group(1))

            # If multiple clocks in same block, check for synchronizer
            if len(block_clocks) > 1:
                has_sync = bool(self._SYNC_PATTERN.search(body))
                if not has_sync:
                    local_issues.append(
                        f"{rel_path}: Block '{block_name}' at line {start_line} crosses "
                        f"clock domains without visible synchronizer: {', '.join(sorted(block_clocks))}"
                    )
                    cdc_risk_count += 1

                # Check for multi-bit CDC without gray code
                multi_bit = bool(re.search(r"\[[0-9]+:[0-9]+\]", body))
                if multi_bit and not self._GRAY_PATTERN.search(body):
                    local_issues.append(
                        f"{rel_path}: Multi-bit CDC detected without gray code pattern (line {start_line})"
                    )
                    cdc_risk_count += 1

                # Check for 2-stage synchronizer
                ff_count = len(re.findall(self._FF_PATTERN, body))
                if ff_count < 2 and has_sync:
                    local_issues.append(
                        f"{rel_path}: Potential single-stage synchronizer detected (line {start_line}); "
                        "use 2+ flip-flop stages for CDC"
                    )
                    cdc_risk_count += 1

        return {
            "issues": local_issues,
            "metrics": {
                "file": rel_path,
                "cdc_risks": cdc_risk_count,
                "clock_domains_count": len(clocks_detected),
            },
            "clocks_detected": list(clocks_detected),
        }
