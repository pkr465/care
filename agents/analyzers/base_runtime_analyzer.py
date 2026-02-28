import re
from typing import List, Tuple, Dict, Any, Optional

class RuntimeAnalyzerBase:
    """
    Base class for Verilog/SystemVerilog runtime analyzers providing shared utilities for
    parsing module blocks, always blocks, and standardizing the analysis interface.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    def analyze(self, file_cache: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for batch analysis of a codebase.
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def analyze_single_file(self, source: str, rel_path: str) -> Dict[str, Any]:
        """
        Analyze a single file's content in isolation.
        Returns a dictionary containing 'issues' (list) and 'metrics' (dict).
        """
        raise NotImplementedError("Subclasses must implement analyze_single_file()")

    def _get_function_blocks(self, source: str) -> List[Tuple[str, str, int]]:
        """
        Splits Verilog/SystemVerilog source into block structures (module, always, task, function, initial).
        Returns list of (block_name, body_content, start_line).

        Heuristic: Tracks begin/end balance or module/endmodule balance to identify block bodies.
        """
        blocks = []
        lines = source.splitlines()
        current_block = None
        current_body = []
        depth = 0
        start_line = 0

        # Regex patterns for block headers
        module_pattern = re.compile(r"^\s*module\s+([a-zA-Z0-9_]+)\s*[#\(]")
        always_pattern = re.compile(r"^\s*always\s*(@|\*)")
        always_ff_pattern = re.compile(r"^\s*always_ff\s*@")
        always_comb_pattern = re.compile(r"^\s*always_comb\b")
        initial_pattern = re.compile(r"^\s*initial\b")
        task_pattern = re.compile(r"^\s*task\s+([a-zA-Z0-9_]+)")
        function_pattern = re.compile(r"^\s*function\s+(?:\w+\s+)?([a-zA-Z0-9_]+)")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip if already inside a block
            if depth == 0:
                # Try to match block headers
                match_module = module_pattern.match(line)
                match_always = always_pattern.match(line)
                match_always_ff = always_ff_pattern.match(line)
                match_always_comb = always_comb_pattern.match(line)
                match_initial = initial_pattern.match(line)
                match_task = task_pattern.match(line)
                match_function = function_pattern.match(line)

                if match_module:
                    current_block = f"module {match_module.group(1)}"
                    current_body = [line]
                    depth = line.count("(") - line.count(")")
                    if depth == 0:  # Port list might be on same line or following
                        depth = 1  # Will be closed by 'endmodule'
                    start_line = i + 1
                elif match_always or match_always_ff or match_always_comb:
                    current_block = "always_block"
                    current_body = [line]
                    depth = 1
                    start_line = i + 1
                elif match_initial:
                    current_block = "initial_block"
                    current_body = [line]
                    depth = 1
                    start_line = i + 1
                elif match_task:
                    current_block = f"task {match_task.group(1)}"
                    current_body = [line]
                    depth = 1
                    start_line = i + 1
                elif match_function:
                    current_block = f"function {match_function.group(1)}"
                    current_body = [line]
                    depth = 1
                    start_line = i + 1
            else:
                # Inside a block, track depth
                current_body.append(line)

                # Track begin/end and module/endmodule
                begins = line.count("begin")
                ends = line.count("end")
                modules = line.count("module")
                endmodules = line.count("endmodule")

                depth += (begins - ends + modules - endmodules)

                # Check for end keywords
                if "endmodule" in line or "endtask" in line or "endfunction" in line or "end" in line:
                    if depth <= 0:
                        # End of block
                        blocks.append((current_block, "\n".join(current_body), start_line))
                        depth = 0
                        current_block = None
                        current_body = []

            i += 1

        return blocks
