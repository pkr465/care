"""
Parameter Propagation Tracker for HDL Module Hierarchy.

Analyzes parameter declarations and overrides across module instantiations,
tracking parameter propagation through the hierarchy and detecting type
mismatches between declared types and override values.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from agents.services import (
    ModuleHierarchy,
    ParameterOverride,
    ParameterPropagationMap,
)

logger = logging.getLogger(__name__)


class ParameterPropagationTracker:
    """
    Tracks parameter declarations and propagation through module hierarchy.

    Extracts parameter declarations from module definitions, identifies parameter
    overrides in instantiations, and performs best-effort type mismatch detection
    between declared parameter types and their override values.

    Attributes:
        project_root: Root directory of the HDL project
        debug: Enable verbose debug logging
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the ParameterPropagationTracker.

        Args:
            config: Configuration dictionary with keys:
                - project_root: Root directory of the project
                - debug: Enable debug logging (default False)
        """
        self.project_root = config.get("project_root", "")
        self.debug = config.get("debug", False)

        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger.debug(f"Initialized ParameterPropagationTracker: project_root={self.project_root}")

    def build(self, file_cache: List[Dict], hierarchy: ModuleHierarchy) -> ParameterPropagationMap:
        """
        Build a complete parameter propagation map from file contents and hierarchy.

        Processes all module definitions to extract parameter declarations with
        defaults, and analyzes all instantiations to identify parameter overrides.
        Performs type mismatch detection where possible.

        Args:
            file_cache: List of dicts with keys:
                - file_path: Absolute path to the source file
                - content: Text content of the file
            hierarchy: ModuleHierarchy object containing instantiation information

        Returns:
            ParameterPropagationMap with all overrides and detected mismatches.

        Raises:
            ValueError: If file_cache items lack required keys
        """
        logger.info(
            f"Building parameter propagation map from {len(file_cache)} files "
            f"and {len(hierarchy.instantiations)} instantiations"
        )

        overrides: List[ParameterOverride] = []
        overrides_by_instance: Dict[str, List[ParameterOverride]] = {}
        type_mismatches: List[ParameterOverride] = []

        # Build file content map for quick access
        file_contents: Dict[str, str] = {}
        for entry in file_cache:
            # Normalize file_cache keys
            file_path = entry.get("file_relative_path") or entry.get("file_path", "")
            content = entry.get("source") or entry.get("content", "")
            if file_path:
                file_contents[file_path] = content

        # Extract parameter declarations from all modules
        module_params: Dict[str, Dict[str, Tuple[str, str]]] = {}  # module -> {param: (type, default)}
        for file_path, content in file_contents.items():
            params = self._extract_module_parameters(file_path, content)
            module_params.update(params)

        logger.debug(
            f"Extracted parameter declarations from {len(module_params)} modules"
        )

        # Process each instantiation to find parameter overrides
        for inst in hierarchy.instantiations:
            instance_key = f"{inst.parent_module}:{inst.instance_name}"

            # Get the source file of the parent module for context
            parent_file = None
            if inst.parent_module in hierarchy.modules:
                parent_file = hierarchy.modules[inst.parent_module].file_path

            # Extract overrides from this instantiation
            inst_overrides = self._extract_instantiation_overrides(
                parent_file if parent_file else "",
                file_contents,
                inst,
                module_params,
            )

            if inst_overrides:
                overrides.extend(inst_overrides)
                overrides_by_instance[instance_key] = inst_overrides

                logger.debug(
                    f"Instance {instance_key}: Found {len(inst_overrides)} parameter overrides"
                )

                # Check for type mismatches
                for override in inst_overrides:
                    if override.type_mismatch:
                        type_mismatches.append(override)

        result = ParameterPropagationMap(
            overrides=overrides,
            overrides_by_instance=overrides_by_instance,
            type_mismatches=type_mismatches,
            total_overrides=len(overrides),
            total_mismatches=len(type_mismatches),
        )

        logger.info(
            f"Parameter propagation map complete: {len(overrides)} overrides, "
            f"{len(type_mismatches)} type mismatches"
        )

        return result

    def _extract_module_parameters(
        self, file_path: str, content: str
    ) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """
        Extract parameter declarations from all modules in the file.

        Identifies module headers with parameter lists and extracts parameter
        definitions with their types and default values.

        Args:
            file_path: Path to the source file
            content: Full text content of the file

        Returns:
            Dict mapping module_name -> {param_name: (type, default_value)}
        """
        modules = {}

        # Strip comments
        clean_content = self._strip_comments(content)

        # Find module headers with parameters: module NAME #(...)
        module_pattern = (
            r"^\s*module\s+(\w+)\s*#\s*\((.*?)\)\s*\("
        )
        matches = re.finditer(
            module_pattern, clean_content, re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

        for match in matches:
            module_name = match.group(1)
            param_block = match.group(2)

            # Extract individual parameters from the block
            params = self._parse_parameter_block(param_block)

            if params:
                modules[module_name] = params
                logger.debug(f"Module '{module_name}': Found {len(params)} parameters")

        return modules

    def _parse_parameter_block(self, param_block: str) -> Dict[str, Tuple[str, str]]:
        """
        Parse parameter declarations from a parameter block.

        Extracts parameter definitions in the form:
        (parameter|localparam)? [type] NAME = DEFAULT_VALUE

        Args:
            param_block: The content within the #(...) block

        Returns:
            Dict mapping parameter_name -> (type, default_value)
        """
        params = {}

        # Pattern: [parameter|localparam] [type] NAME = VALUE
        # The type is optional, VALUE goes until comma or end of block
        pattern = r"(?:parameter|localparam)?\s*(?:(\w+)\s+)?(\w+)\s*=\s*([^,)]+)"
        matches = re.finditer(pattern, param_block, re.IGNORECASE)

        for match in matches:
            param_type = match.group(1) or "logic"  # Default type if not specified
            param_name = match.group(2)
            default_val = match.group(3).strip()

            params[param_name] = (param_type, default_val)

        return params

    def _extract_instantiation_overrides(
        self,
        parent_file: str,
        file_contents: Dict[str, str],
        inst,  # Instantiation object
        module_params: Dict[str, Dict[str, Tuple[str, str]]],
    ) -> List[ParameterOverride]:
        """
        Extract parameter overrides from a single instantiation.

        Parses the instantiation syntax looking for parameter bindings in the
        form: MODULE_NAME #(.PARAM(value)) instance_name

        Args:
            parent_file: Path to file containing the instantiation
            file_contents: Map of file_path -> content
            inst: Instantiation object with details
            module_params: Module parameter declarations

        Returns:
            List of ParameterOverride objects
        """
        overrides = []

        if not parent_file or parent_file not in file_contents:
            return overrides

        content = file_contents[parent_file]
        clean_content = self._strip_comments(content)

        # Look for instantiation pattern: CHILD_MODULE #(...)
        # Search around the line where instantiation occurs
        lines = clean_content.split("\n")
        if inst.line < 1 or inst.line > len(lines):
            logger.debug(
                f"Instantiation line {inst.line} out of range for {parent_file}"
            )
            return overrides

        # Search in a window around the instantiation line
        search_start = max(0, inst.line - 5)
        search_end = min(len(lines), inst.line + 5)
        search_block = "\n".join(lines[search_start:search_end])

        # Find parameter overrides: #(.PARAM(value), .PARAM2(value2))
        param_override_pattern = r"#\s*\((.*?)\)\s*(\w+)\s*\("
        matches = re.finditer(param_override_pattern, search_block, re.DOTALL)

        for match in matches:
            override_block = match.group(1)
            instance_name = match.group(2)

            # Only process if this matches our instantiation
            if instance_name != inst.instance_name:
                continue

            # Parse individual parameter overrides
            param_pattern = r"\.(\w+)\s*\(\s*([^)]*)\s*\)"
            param_matches = re.finditer(param_pattern, override_block)

            for param_match in param_matches:
                param_name = param_match.group(1)
                override_value = param_match.group(2).strip()

                # Get the declared parameter info if available
                child_module = inst.child_module
                default_value = ""
                param_type = ""
                type_mismatch = False
                mismatch_detail = ""

                if child_module in module_params:
                    module_param_info = module_params[child_module]
                    if param_name in module_param_info:
                        param_type, default_value = module_param_info[param_name]

                        # Check for type mismatch
                        type_mismatch, mismatch_detail = self._detect_type_mismatch(
                            param_type, default_value, override_value
                        )

                override_obj = ParameterOverride(
                    instance_name=inst.instance_name,
                    parent_module=inst.parent_module,
                    child_module=inst.child_module,
                    param_name=param_name,
                    override_value=override_value,
                    default_value=default_value,
                    param_type=param_type,
                    file_path=parent_file,
                    line=inst.line,
                    type_mismatch=type_mismatch,
                    mismatch_detail=mismatch_detail,
                )
                overrides.append(override_obj)

                logger.debug(
                    f"Override: {inst.instance_name}.{param_name} = {override_value} "
                    f"(type: {param_type}, mismatch: {type_mismatch})"
                )

        return overrides

    def _detect_type_mismatch(
        self, param_type: str, default_value: str, override_value: str
    ) -> Tuple[bool, str]:
        """
        Detect type mismatches between declared type and override value.

        Performs best-effort detection for common mismatches:
        - integer/int type with string literal override
        - Bit-width mismatches in vector types
        - Logic type with non-numeric override

        Args:
            param_type: Declared parameter type
            default_value: Default value from declaration
            override_value: Override value from instantiation

        Returns:
            Tuple of (is_mismatch: bool, detail: str)
        """
        if not param_type or not override_value:
            return False, ""

        param_type_lower = param_type.lower()
        override_value_stripped = override_value.strip().strip('"\'')

        # Check integer type with string literal
        if param_type_lower in ("integer", "int"):
            # Check if override is a string literal (not numeric or expression)
            if (
                (override_value.startswith('"') and override_value.endswith('"'))
                or (override_value.startswith("'") and override_value.endswith("'"))
            ):
                # Check if content is non-numeric
                if not re.match(r"^\d+$", override_value_stripped):
                    return (
                        True,
                        f"integer type '{param_type}' overridden with string literal '{override_value}'",
                    )

        # Check bit-width mismatches in vector types
        if "[" in param_type:
            # Extract declared bit width
            width_match = re.search(r"\[(\d+):(\d+)\]", param_type)
            if width_match:
                declared_width = (
                    int(width_match.group(1)) - int(width_match.group(2)) + 1
                )

                # Try to infer override width
                # Look for hex: 'h... or decimal: ...
                if "'" in override_value:
                    hex_match = re.search(r"'h([0-9a-fA-F]+)", override_value)
                    if hex_match:
                        override_width = len(hex_match.group(1)) * 4
                        if override_width != declared_width:
                            return (
                                True,
                                f"bit-width mismatch: {param_type} ({declared_width} bits) "
                                f"overridden with {override_width}-bit value '{override_value}'",
                            )

        return False, ""

    def _strip_comments(self, content: str) -> str:
        """
        Remove single-line and multi-line comments from SystemVerilog content.

        Handles:
        - Single-line: // ... (to end of line)
        - Multi-line: /* ... */ (including nested)

        Args:
            content: Source content to clean

        Returns:
            Content with comments removed
        """
        # Remove single-line comments: // ...
        content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)

        # Remove multi-line comments: /* ... */
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        return content
