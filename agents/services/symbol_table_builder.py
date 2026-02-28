"""
Symbol Table Builder — Cross-File Symbol Resolution and Collision Detection.

Builds a comprehensive symbol table from HDL source files, tracking signals,
parameters, macros, functions, tasks, types, and scope information. Handles
package imports, include propagation, and symbol collision detection.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from . import (
    IncludeTree,
    PackageImportMap,
    SymbolTable,
    SymbolDef,
)

logger = logging.getLogger(__name__)


class SymbolTableBuilder:
    """
    Builds a complete symbol table from HDL files with scope and import resolution.

    Handles:
    - Port and signal extraction with type information
    - Parameter and localparam declarations
    - Macro definitions
    - Function and task declarations
    - Typedef, enum, and struct definitions
    - Scope assignment (module, package, or global)
    - Include-based symbol propagation
    - Package import resolution
    - Symbol collision detection across files
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the symbol table builder.

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

    def _extract_current_scope(self, source: str, position: int) -> str:
        """
        Determine the scope (module or package) at a given position in source.

        Args:
            source: HDL source code
            position: Character position in source

        Returns:
            Scope name (module_name, package_name, or empty string for global)
        """
        prefix = source[:position]

        # Find the most recent module or package declaration
        modules = list(re.finditer(r"^\s*module\s+(\w+)", prefix, re.MULTILINE))
        packages = list(re.finditer(r"^\s*package\s+(\w+)", prefix, re.MULTILINE))

        # Get the last (most recent) module or package
        scopes = []
        if modules:
            scopes.append(("module", modules[-1].group(1), modules[-1].start()))
        if packages:
            scopes.append(("package", packages[-1].group(1), packages[-1].start()))

        if scopes:
            # Return the one with the latest position
            scope_type, scope_name, scope_pos = max(scopes, key=lambda x: x[2])
            return scope_name

        return ""

    def _extract_ports(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract port declarations from module or interface.

        Pattern: `(input|output|inout|ref) [type] [width] name`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope (module or package name)

        Returns:
            List of SymbolDef objects for ports
        """
        ports: List[SymbolDef] = []

        # Match port declarations within module declaration
        # Simplified: look for port declarations in module header
        port_pattern = (
            r"(input|output|inout|ref)\s+(?:(wire|reg|logic|bit|integer|real|time|string))?"
            r"\s*(?:signed|unsigned)?\s*(?:\[.*?\])?\s*(\w+)"
        )

        matches = re.finditer(port_pattern, source)

        for match in matches:
            direction = match.group(1)
            data_type = match.group(2) or "logic"
            name = match.group(3)
            line_no = source[: match.start()].count("\n") + 1

            ports.append(
                SymbolDef(
                    name=name,
                    kind="port",
                    data_type=data_type,
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                    value=direction,  # Store direction in value field
                )
            )

        return ports

    def _extract_signals(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract signal declarations.

        Pattern: `(wire|reg|logic|bit) [signed|unsigned] [width] name`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for signals
        """
        signals: List[SymbolDef] = []

        # Match signal declarations
        signal_pattern = (
            r"(wire|reg|logic|bit)\s+(?:signed|unsigned)?\s*(?:\[.*?\])?\s*(\w+)"
        )

        matches = re.finditer(signal_pattern, source)

        for match in matches:
            signal_type = match.group(1)
            name = match.group(2)
            line_no = source[: match.start()].count("\n") + 1

            signals.append(
                SymbolDef(
                    name=name,
                    kind="signal",
                    data_type=signal_type,
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                )
            )

        return signals

    def _extract_parameters(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract parameter and localparam declarations.

        Pattern: `(parameter|localparam) [type] name = value`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for parameters
        """
        parameters: List[SymbolDef] = []

        # Match parameter declarations
        param_pattern = (
            r"(parameter|localparam)\s+(?:(\w+)\s+)?(\w+)\s*=\s*([^;,\n]+)"
        )

        matches = re.finditer(param_pattern, source)

        for match in matches:
            param_type = match.group(1)
            data_type = match.group(2) or ""
            name = match.group(3)
            default_value = match.group(4).strip()
            line_no = source[: match.start()].count("\n") + 1

            is_localparam = param_type == "localparam"

            parameters.append(
                SymbolDef(
                    name=name,
                    kind="parameter",
                    data_type=data_type,
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                    value=default_value,
                )
            )

        return parameters

    def _extract_macros(self, source: str, file_path: str) -> List[SymbolDef]:
        """
        Extract macro definitions.

        Pattern: `` `define name(args)? value ``

        Macros are always global scope.

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file

        Returns:
            List of SymbolDef objects for macros
        """
        macros: List[SymbolDef] = []

        # Match macro definitions
        macro_pattern = r"`define\s+(\w+)(?:\(.*?\))?\s+(.*)$"

        matches = re.finditer(macro_pattern, source, re.MULTILINE)

        for match in matches:
            name = match.group(1)
            value = match.group(2).strip()
            line_no = source[: match.start()].count("\n") + 1

            macros.append(
                SymbolDef(
                    name=name,
                    kind="macro",
                    file_path=file_path,
                    line=line_no,
                    scope="",  # Global
                    value=value,
                )
            )

        return macros

    def _extract_functions(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract function declarations.

        Pattern: `function [automatic] [return_type] name (...)`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for functions
        """
        functions: List[SymbolDef] = []

        # Match function declarations
        func_pattern = (
            r"function\s+(?:automatic\s+)?(?:(\w+)\s+)?(\w+)\s*\("
        )

        matches = re.finditer(func_pattern, source)

        for match in matches:
            return_type = match.group(1) or ""
            name = match.group(2)
            line_no = source[: match.start()].count("\n") + 1

            functions.append(
                SymbolDef(
                    name=name,
                    kind="function",
                    data_type=return_type,
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                )
            )

        return functions

    def _extract_tasks(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract task declarations.

        Pattern: `task [automatic] name (...)`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for tasks
        """
        tasks: List[SymbolDef] = []

        # Match task declarations
        task_pattern = r"task\s+(?:automatic\s+)?(\w+)\s*\("

        matches = re.finditer(task_pattern, source)

        for match in matches:
            name = match.group(1)
            line_no = source[: match.start()].count("\n") + 1

            tasks.append(
                SymbolDef(
                    name=name,
                    kind="task",
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                )
            )

        return tasks

    def _extract_typedefs(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract typedef declarations (generic typedefs, not struct/enum).

        Pattern: `typedef ... name;`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for typedefs
        """
        typedefs: List[SymbolDef] = []

        # Match simple typedef (not struct/union/enum which are handled separately)
        # Match: typedef base_type new_type_name;
        typedef_pattern = r"typedef\s+(?!(?:struct|union|enum))(?:\w+\s+)*(\w+)\s*;"

        matches = re.finditer(typedef_pattern, source)

        for match in matches:
            name = match.group(1)
            line_no = source[: match.start()].count("\n") + 1

            typedefs.append(
                SymbolDef(
                    name=name,
                    kind="type",
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                )
            )

        return typedefs

    def _extract_enums(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract enum type declarations.

        Pattern: `typedef enum ... name;`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for enums
        """
        enums: List[SymbolDef] = []

        # Match enum typedefs: typedef enum {...} name;
        enum_pattern = r"typedef\s+enum\s+.*?\}\s*(\w+)\s*;"

        matches = re.finditer(enum_pattern, source, re.DOTALL)

        for match in matches:
            name = match.group(1)
            line_no = source[: match.start()].count("\n") + 1

            enums.append(
                SymbolDef(
                    name=name,
                    kind="enum",
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                )
            )

        return enums

    def _extract_structs(self, source: str, file_path: str, scope: str) -> List[SymbolDef]:
        """
        Extract struct/union type declarations.

        Pattern: `typedef [packed] struct/union {...} name;`

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of SymbolDef objects for structs
        """
        structs: List[SymbolDef] = []

        # Match struct/union typedefs: typedef [packed] struct/union {...} name;
        struct_pattern = r"typedef\s+(?:packed\s+)?(?:struct|union)\s+.*?\}\s*(\w+)\s*;"

        matches = re.finditer(struct_pattern, source, re.DOTALL)

        for match in matches:
            name = match.group(1)
            line_no = source[: match.start()].count("\n") + 1

            structs.append(
                SymbolDef(
                    name=name,
                    kind="struct",
                    file_path=file_path,
                    line=line_no,
                    scope=scope,
                )
            )

        return structs

    def _extract_all_symbols(
        self, source: str, file_path: str, scope: str
    ) -> List[SymbolDef]:
        """
        Extract all symbols from a file.

        Args:
            source: HDL source code (comment-stripped)
            file_path: Path to the file
            scope: Current scope

        Returns:
            List of all SymbolDef objects found
        """
        symbols: List[SymbolDef] = []

        symbols.extend(self._extract_ports(source, file_path, scope))
        symbols.extend(self._extract_signals(source, file_path, scope))
        symbols.extend(self._extract_parameters(source, file_path, scope))
        symbols.extend(self._extract_macros(source, file_path))
        symbols.extend(self._extract_functions(source, file_path, scope))
        symbols.extend(self._extract_tasks(source, file_path, scope))
        symbols.extend(self._extract_typedefs(source, file_path, scope))
        symbols.extend(self._extract_enums(source, file_path, scope))
        symbols.extend(self._extract_structs(source, file_path, scope))

        return symbols

    def _get_module_scope(self, source: str) -> str:
        """
        Extract the module name from source (scope).

        Args:
            source: HDL source code

        Returns:
            Module name or empty string
        """
        match = re.search(r"^\s*module\s+(\w+)", source, re.MULTILINE)
        return match.group(1) if match else ""

    def build(
        self,
        file_cache: List[Dict],
        include_tree: IncludeTree,
        package_map: PackageImportMap,
    ) -> SymbolTable:
        """
        Build complete symbol table from file cache with import resolution.

        Args:
            file_cache: List of file dicts with keys:
                - path: file path
                - source: HDL source code
            include_tree: Include dependency tree for symbol propagation
            package_map: Package import map for symbol resolution

        Returns:
            SymbolTable with all resolved symbols and collisions
        """
        result = SymbolTable()

        # Per-file symbol extraction
        file_symbols: Dict[str, List[SymbolDef]] = {}

        for file_entry in file_cache:
            # Normalize file_cache keys
            file_path = file_entry.get("file_relative_path") or file_entry.get("file_path") or file_entry.get("path", "")
            source = file_entry.get("source", "")

            if not source:
                continue

            # Strip comments
            clean_source = self._strip_comments(source)

            # Determine scope (module or package name)
            scope = self._get_module_scope(clean_source)

            if self.debug:
                self.logger.debug(f"Analyzing {file_path} (scope: {scope})")

            # Extract all symbols from this file
            symbols = self._extract_all_symbols(clean_source, file_path, scope)
            file_symbols[file_path] = symbols

            # Add symbols to table (collision detection happens in add())
            for sym in symbols:
                result.add(sym)

        # Include propagation: add symbols from transitive includes
        for file_path, symbol_list in file_symbols.items():
            transitive_includes = include_tree.include_chains.get(file_path, [])

            for include_path in transitive_includes:
                included_symbols = file_symbols.get(include_path, [])
                for sym in included_symbols:
                    # Create a new symbol def with reference to included file
                    result.add(sym)

        # Package import resolution
        for file_path, import_statements in package_map.imports_by_file.items():
            for import_stmt in import_statements:
                package_name = import_stmt.package_name
                symbol_name = import_stmt.symbol

                # Look up package definition
                package_def = package_map.package_defs.get(package_name)
                if not package_def:
                    continue

                if symbol_name == "*":
                    # Wildcard import: add all exported symbols from package
                    for exported_symbol in package_def.exported_symbols:
                        # Create symbol with package scope
                        sym = SymbolDef(
                            name=exported_symbol,
                            kind="parameter",  # or other kind
                            file_path=package_def.file_path,
                            scope=package_name,
                        )
                        result.add(sym)
                else:
                    # Specific import: add just that symbol
                    if symbol_name in package_def.exported_symbols:
                        sym = SymbolDef(
                            name=symbol_name,
                            kind="parameter",
                            file_path=package_def.file_path,
                            scope=package_name,
                        )
                        result.add(sym)

        if self.debug:
            self.logger.debug(
                f"Symbol table complete: {result.total_symbols} symbols, "
                f"{result.total_collisions} collisions"
            )

        return result
