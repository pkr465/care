"""
Generate Block Expander — SystemVerilog Conditional Generation Analysis.

Analyzes generate blocks (if/for/case) in HDL modules and tracks conditional
module instantiations. Identifies which modules are conditionally instantiated
based on generate conditions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from . import GenerateBlock, GenerateBlockExpansions, ModuleHierarchy

logger = logging.getLogger(__name__)


class GenerateBlockExpander:
    """
    Analyzes HDL files for generate blocks and conditional instantiations.

    Detects:
    - Explicit generate...endgenerate blocks
    - Generate-if blocks with optional labels
    - Generate-for blocks with loop variables
    - Generate-case blocks with conditions
    - Module instantiations within generate blocks
    - Mapping of conditionally-instantiated modules to their conditions
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the expander.

        Args:
            config: Configuration dict with keys:
                - project_root: Root directory of the HDL project
                - debug: Whether to log verbose debug messages
        """
        self.project_root = config.get("project_root", "")
        self.debug = config.get("debug", False)
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

    def _strip_comments(self, source: str) -> str:
        """
        Remove SystemVerilog comments from source code.

        Strips single-line (//) and multi-line (/* */) comments.

        Args:
            source: HDL source code

        Returns:
            Source code with comments removed
        """
        # Remove multi-line comments
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        # Remove single-line comments
        source = re.sub(r"//.*?$", "", source, flags=re.MULTILINE)
        return source

    def _extract_module_name_from_source(self, source: str) -> Optional[str]:
        """
        Extract the module name from HDL source.

        Args:
            source: HDL source code

        Returns:
            Module name or None
        """
        match = re.search(r"^\s*module\s+(\w+)", source, re.MULTILINE)
        return match.group(1) if match else None

    def _find_balanced_parentheses(self, text: str, start: int) -> int:
        """
        Find the closing parenthesis matching the opening one at start.

        Args:
            text: Source text
            start: Index of opening parenthesis

        Returns:
            Index of matching closing parenthesis, or -1 if not found
        """
        if start >= len(text) or text[start] != "(":
            return -1

        depth = 1
        i = start + 1
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1

        return i - 1 if depth == 0 else -1

    def _extract_explicit_generate_blocks(self, source: str) -> List[Dict[str, Any]]:
        """
        Extract explicit generate...endgenerate blocks.

        Args:
            source: Comment-stripped HDL source code

        Returns:
            List of dicts with block info: type, condition, body, line, label
        """
        blocks = []

        # Match generate...endgenerate blocks
        pattern = r"^\s*generate\s*\n(.*?)^\s*endgenerate"
        matches = re.finditer(pattern, source, re.MULTILINE | re.DOTALL)

        for match in matches:
            body = match.group(1)
            line_no = source[: match.start()].count("\n") + 1

            blocks.append(
                {
                    "type": "explicit",
                    "body": body,
                    "line": line_no,
                    "condition": "",
                    "label": "",
                }
            )

        return blocks

    def _extract_generate_if_blocks(self, source: str) -> List[Dict[str, Any]]:
        """
        Extract generate-if blocks.

        Pattern: `if (condition) begin [:label] ... end`

        These are implicit generate blocks when at module level.

        Args:
            source: Comment-stripped HDL source code

        Returns:
            List of dicts with block info
        """
        blocks = []

        # Match generate-if: if (condition) begin ... end
        # with optional label
        pattern = r"^\s*if\s*\((.*?)\)\s*begin\s*(?::(\w+))?\s*\n(.*?)^\s*end"
        matches = re.finditer(pattern, source, re.MULTILINE | re.DOTALL)

        for match in matches:
            condition = match.group(1).strip()
            label = match.group(2) or ""
            body = match.group(3)
            line_no = source[: match.start()].count("\n") + 1

            # Skip if this is inside an always/initial/task/function block
            if self._is_inside_procedural_block(source, match.start()):
                continue

            blocks.append(
                {
                    "type": "if",
                    "condition": condition,
                    "body": body,
                    "line": line_no,
                    "label": label,
                }
            )

        return blocks

    def _extract_generate_for_blocks(self, source: str) -> List[Dict[str, Any]]:
        """
        Extract generate-for blocks.

        Pattern: `for (init; condition; increment) begin [:label] ... end`

        Args:
            source: Comment-stripped HDL source code

        Returns:
            List of dicts with block info
        """
        blocks = []

        # Match generate-for
        pattern = r"^\s*for\s*\((.*?)\)\s*begin\s*(?::(\w+))?\s*\n(.*?)^\s*end"
        matches = re.finditer(pattern, source, re.MULTILINE | re.DOTALL)

        for match in matches:
            condition = match.group(1).strip()
            label = match.group(2) or ""
            body = match.group(3)
            line_no = source[: match.start()].count("\n") + 1

            # Skip if inside procedural block
            if self._is_inside_procedural_block(source, match.start()):
                continue

            blocks.append(
                {
                    "type": "for",
                    "condition": condition,
                    "body": body,
                    "line": line_no,
                    "label": label,
                }
            )

        return blocks

    def _extract_generate_case_blocks(self, source: str) -> List[Dict[str, Any]]:
        """
        Extract generate-case blocks.

        Pattern: `case (expression) ... endcase`

        Args:
            source: Comment-stripped HDL source code

        Returns:
            List of dicts with block info
        """
        blocks = []

        # Match generate-case
        pattern = r"^\s*case\s*\((.*?)\)\s*\n(.*?)^\s*endcase"
        matches = re.finditer(pattern, source, re.MULTILINE | re.DOTALL)

        for match in matches:
            condition = match.group(1).strip()
            body = match.group(2)
            line_no = source[: match.start()].count("\n") + 1

            # Skip if inside procedural block
            if self._is_inside_procedural_block(source, match.start()):
                continue

            blocks.append(
                {
                    "type": "case",
                    "condition": condition,
                    "body": body,
                    "line": line_no,
                    "label": "",
                }
            )

        return blocks

    def _is_inside_procedural_block(self, source: str, position: int) -> bool:
        """
        Check if position is inside an always/initial/task/function block.

        Args:
            source: HDL source code
            position: Character position

        Returns:
            True if inside a procedural block, False otherwise
        """
        # Simple heuristic: look backwards for always/initial/task/function
        # and check if we're in its scope
        prefix = source[:position]

        # Count unmatched begin/end to estimate scope
        always_tasks = re.findall(
            r"\b(always|initial|task|function|always_ff|always_comb|always_latch)\b",
            prefix,
        )
        begins = prefix.count("begin")
        ends = prefix.count("end")

        # If we have more begins than ends after an always/task, we're inside
        if always_tasks and begins > ends:
            return True

        return False

    def _extract_instantiations_from_body(
        self, body: str, parent_module: str, file_path: str, condition: str
    ) -> tuple[List[str], List[GenerateBlock]]:
        """
        Extract module instantiations from generate block body.

        Pattern: `module_name instance_name (...)`

        Args:
            body: Generate block body source
            parent_module: Name of parent module
            file_path: Path to source file
            condition: The generate condition expression

        Returns:
            Tuple of (instance names, GenerateBlock objects)
        """
        instances = []
        generate_blocks = []

        # Match module instantiations
        # Pattern: word_word ( ... ) or word_word #(...) (...)
        pattern = r"(\w+)\s+(\w+)\s*(?:#\s*\(.*?\))?\s*\("
        matches = re.finditer(pattern, body, re.DOTALL)

        for match in matches:
            module_type = match.group(1)
            instance_name = match.group(2)
            instances.append(module_type)

        return instances, generate_blocks

    def build(
        self, file_cache: List[Dict], hierarchy: ModuleHierarchy
    ) -> GenerateBlockExpansions:
        """
        Build complete generate block expansion map from file cache.

        Args:
            file_cache: List of file dicts with keys:
                - file_relative_path or file_path: file path
                - source: HDL source code
            hierarchy: Module hierarchy for reference

        Returns:
            GenerateBlockExpansions with all generate blocks and conditional instances
        """
        result = GenerateBlockExpansions()

        blocks_by_module: Dict[str, List[GenerateBlock]] = {}
        all_blocks: List[GenerateBlock] = []
        conditional_instances: Dict[str, str] = {}

        for file_entry in file_cache:
            # Normalize file_cache keys
            file_path = file_entry.get("file_relative_path") or file_entry.get("file_path") or file_entry.get("path", "")
            source = file_entry.get("source", "")

            if not source:
                continue

            # Strip comments
            clean_source = self._strip_comments(source)

            # Extract module name
            module_name = self._extract_module_name_from_source(clean_source)
            if not module_name:
                continue

            if self.debug:
                self.logger.debug(f"Analyzing {file_path} (module: {module_name})")

            # Find all generate blocks in this module
            generate_blocks = []

            # Extract explicit generate...endgenerate blocks
            explicit = self._extract_explicit_generate_blocks(clean_source)
            generate_blocks.extend(explicit)

            # Extract implicit generate blocks (if/for/case at module level)
            generate_blocks.extend(self._extract_generate_if_blocks(clean_source))
            generate_blocks.extend(self._extract_generate_for_blocks(clean_source))
            generate_blocks.extend(self._extract_generate_case_blocks(clean_source))

            # Process each generate block
            for block_info in generate_blocks:
                block_type = block_info["type"]
                body = block_info["body"]
                line_no = block_info["line"]
                condition = block_info["condition"]
                label = block_info["label"]

                # Extract instantiations from this block
                contained_instances, _ = self._extract_instantiations_from_body(
                    body, module_name, file_path, condition
                )

                # Create GenerateBlock record
                gen_block = GenerateBlock(
                    block_type=block_type,
                    condition=condition,
                    parent_module=module_name,
                    file_path=file_path,
                    line=line_no,
                    contained_instances=contained_instances,
                    label=label,
                )

                all_blocks.append(gen_block)

                # Track by module
                if module_name not in blocks_by_module:
                    blocks_by_module[module_name] = []
                blocks_by_module[module_name].append(gen_block)

                # Record conditional instances
                for instance in contained_instances:
                    conditional_instances[instance] = condition

                if self.debug:
                    self.logger.debug(
                        f"  Found generate-{block_type} in {module_name} "
                        f"(line {line_no}, {len(contained_instances)} instances)"
                    )

        result.blocks = all_blocks
        result.blocks_by_module = blocks_by_module
        result.conditional_instances = conditional_instances
        result.total_generate_blocks = len(all_blocks)
        result.total_conditional_instances = len(conditional_instances)

        if self.debug:
            self.logger.debug(
                f"Generate block expansion complete: "
                f"{result.total_generate_blocks} blocks, "
                f"{result.total_conditional_instances} conditional instances"
            )

        return result
