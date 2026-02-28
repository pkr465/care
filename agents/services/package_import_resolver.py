"""
Package Import Resolver for SystemVerilog HDL Analysis.

Analyzes SystemVerilog source files to extract package definitions and import
statements, building a complete map of package-to-symbol relationships and
resolving unresolved package references.
"""

import logging
import re
from typing import Dict, List, Optional, Set

from agents.services import (
    ImportStatement,
    PackageDefinition,
    PackageImportMap,
)

logger = logging.getLogger(__name__)

# System packages to exclude from resolution
SYSTEM_PACKAGES = {
    "std",
    "ieee",
    "uvm",
    "uvvm",
    "osvvm",
    "xilinx",
    "altera",
    "intel",
    "mentor",
    "cadence",
    "synopsys",
    "vcs",
    "questa",
    "modelsim",
    "vivado",
    "quartus",
}


class PackageImportResolver:
    """
    Resolves package definitions and import statements in SystemVerilog HDL.

    This resolver extracts all package definitions from source files, identifies
    exported symbols (parameters, typedefs, functions, tasks, enums, structs),
    and tracks all import statements. It can optionally exclude system packages
    from resolution.

    Attributes:
        project_root: Root directory of the HDL project
        exclude_system_packages: Whether to exclude standard library packages
        debug: Enable verbose debug logging
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the PackageImportResolver.

        Args:
            config: Configuration dictionary with keys:
                - project_root: Root directory of the project
                - exclude_system_packages: Boolean to filter system packages (default True)
                - debug: Enable debug logging (default False)
        """
        self.project_root = config.get("project_root", "")
        self.exclude_system_packages = config.get("exclude_system_packages", True)
        self.debug = config.get("debug", False)

        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger.debug(
            f"Initialized PackageImportResolver: project_root={self.project_root}, "
            f"exclude_system={self.exclude_system_packages}"
        )

    def build(self, file_cache: List[Dict]) -> PackageImportMap:
        """
        Build a complete package import map from cached file contents.

        Processes all files in the cache to:
        1. Extract package definitions and their exported symbols
        2. Identify all import statements (wildcard and specific)
        3. Filter system packages if configured
        4. Build reverse maps for symbol-to-files and unresolved packages

        Args:
            file_cache: List of dicts with keys:
                - file_relative_path or file_path: Path to the source file
                - source or content: Text content of the file

        Returns:
            PackageImportMap with all package definitions, imports, and symbols.
        """
        logger.info(f"Building package import map from {len(file_cache)} files")

        imports_by_file: Dict[str, List[ImportStatement]] = {}
        package_defs: Dict[str, PackageDefinition] = {}
        symbol_to_files: Dict[str, Set[str]] = {}
        unresolved_packages: Set[str] = set()
        total_imports = 0

        # Process each file
        for file_entry in file_cache:
            # Normalize file_cache keys
            file_path = file_entry.get("file_relative_path") or file_entry.get("file_path", "")
            content = file_entry.get("source") or file_entry.get("content", "")

            # Extract packages in this file
            packages_found = self._extract_packages(file_path, content)
            package_defs.update(packages_found)

            # Extract imports in this file
            imports_found = self._extract_imports(file_path, content)
            if imports_found:
                imports_by_file[file_path] = imports_found
                total_imports += len(imports_found)

                logger.debug(
                    f"{file_path}: Found {len(imports_found)} import statements"
                )

        # Build symbol-to-files map and detect unresolved packages
        for file_path, imports in imports_by_file.items():
            for imp in imports:
                # Check if package is defined in codebase
                if imp.package_name not in package_defs:
                    if not (
                        self.exclude_system_packages
                        and imp.package_name.lower() in SYSTEM_PACKAGES
                    ):
                        unresolved_packages.add(imp.package_name)

                # Track symbol usage
                if imp.symbol != "*":
                    symbol_key = f"{imp.package_name}::{imp.symbol}"
                    if symbol_key not in symbol_to_files:
                        symbol_to_files[symbol_key] = set()
                    symbol_to_files[symbol_key].add(file_path)
                else:
                    # For wildcard imports, track all exported symbols
                    if imp.package_name in package_defs:
                        pkg_def = package_defs[imp.package_name]
                        for symbol in pkg_def.exported_symbols:
                            symbol_key = f"{imp.package_name}::{symbol}"
                            if symbol_key not in symbol_to_files:
                                symbol_to_files[symbol_key] = set()
                            symbol_to_files[symbol_key].add(file_path)

        result = PackageImportMap(
            imports_by_file=imports_by_file,
            package_defs=package_defs,
            symbol_to_files=symbol_to_files,
            unresolved_packages=unresolved_packages,
            total_imports=total_imports,
        )

        logger.info(
            f"Package import map complete: {len(package_defs)} packages, "
            f"{total_imports} imports, {len(unresolved_packages)} unresolved"
        )

        return result

    def _extract_packages(self, file_path: str, content: str) -> Dict[str, PackageDefinition]:
        """
        Extract all package definitions from source content.

        Identifies package...endpackage blocks and extracts exported symbols
        including parameters, typedefs, functions, tasks, enums, and structs.

        Args:
            file_path: Path to the source file
            content: Full text content of the file

        Returns:
            Dict mapping package name to PackageDefinition
        """
        packages = {}

        # Strip comments
        clean_content = self._strip_comments(content)

        # Find all packages: package NAME ... endpackage
        package_pattern = r"^\s*package\s+(\w+)\s*;(.*?)^\s*endpackage\b"
        matches = re.finditer(
            package_pattern, clean_content, re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

        for match in matches:
            pkg_name = match.group(1)
            pkg_body = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            # Extract exported symbols from package body
            parameters = self._extract_parameters(pkg_body)
            typedefs = self._extract_typedefs(pkg_body)
            functions = self._extract_functions(pkg_body)
            tasks = self._extract_tasks(pkg_body)
            enums = self._extract_enums(pkg_body)
            structs = self._extract_structs(pkg_body)

            # Combine all symbols
            all_symbols = set()
            all_symbols.update(parameters)
            all_symbols.update(typedefs)
            all_symbols.update(functions)
            all_symbols.update(tasks)
            all_symbols.update(enums)
            all_symbols.update(structs)

            pkg_def = PackageDefinition(
                name=pkg_name,
                file_path=file_path,
                line=line_num,
                exported_symbols=all_symbols,
                typedefs=typedefs,
                functions=functions,
                tasks=tasks,
            )
            packages[pkg_name] = pkg_def

            logger.debug(
                f"Package '{pkg_name}' in {file_path}: {len(all_symbols)} symbols"
            )

        return packages

    def _extract_imports(self, file_path: str, content: str) -> List[ImportStatement]:
        """
        Extract all import statements from source content.

        Identifies both wildcard imports (import pkg::*;) and specific symbol
        imports (import pkg::symbol;).

        Args:
            file_path: Path to the source file
            content: Full text content of the file

        Returns:
            List of ImportStatement objects
        """
        imports = []

        # Strip comments
        clean_content = self._strip_comments(content)

        # Wildcard imports: import PKG::*;
        wildcard_pattern = r"^\s*import\s+(\w+)\s*::\s*\*\s*;"
        matches = re.finditer(
            wildcard_pattern, clean_content, re.MULTILINE | re.IGNORECASE
        )
        for match in matches:
            pkg_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1
            imports.append(
                ImportStatement(
                    package_name=pkg_name,
                    symbol="*",
                    file_path=file_path,
                    line=line_num,
                )
            )
            logger.debug(f"Found wildcard import: {pkg_name}::* at line {line_num}")

        # Specific imports: import PKG::SYMBOL;
        specific_pattern = r"^\s*import\s+(\w+)\s*::\s*(\w+)\s*;"
        matches = re.finditer(
            specific_pattern, clean_content, re.MULTILINE | re.IGNORECASE
        )
        for match in matches:
            pkg_name = match.group(1)
            symbol = match.group(2)
            line_num = content[: match.start()].count("\n") + 1
            imports.append(
                ImportStatement(
                    package_name=pkg_name,
                    symbol=symbol,
                    file_path=file_path,
                    line=line_num,
                )
            )
            logger.debug(
                f"Found specific import: {pkg_name}::{symbol} at line {line_num}"
            )

        return imports

    def _extract_parameters(self, content: str) -> List[str]:
        """
        Extract parameter and localparam names from content.

        Pattern: (parameter|localparam) [type] NAME [= default]

        Args:
            content: Source content to search

        Returns:
            List of parameter names
        """
        params = []
        pattern = r"(?:parameter|localparam)\s+(?:\w+\s+)?(\w+)\s*="
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            params.append(match.group(1))
        return params

    def _extract_typedefs(self, content: str) -> List[str]:
        """
        Extract typedef names from content.

        Handles: typedef TYPE NAME;

        Args:
            content: Source content to search

        Returns:
            List of typedef names
        """
        typedefs = []
        pattern = r"typedef\s+\w+.*?\s+(\w+)\s*;"
        matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            name = match.group(1)
            # Filter out enum/struct/union keywords
            if name.lower() not in ("enum", "struct", "union"):
                typedefs.append(name)
        return typedefs

    def _extract_functions(self, content: str) -> List[str]:
        """
        Extract function names from content.

        Pattern: function [automatic] [type] NAME (

        Args:
            content: Source content to search

        Returns:
            List of function names
        """
        functions = []
        pattern = r"function\s+(?:automatic\s+)?(?:\w+\s+)?(\w+)\s*\("
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            functions.append(match.group(1))
        return functions

    def _extract_tasks(self, content: str) -> List[str]:
        """
        Extract task names from content.

        Pattern: task [automatic] NAME (

        Args:
            content: Source content to search

        Returns:
            List of task names
        """
        tasks = []
        pattern = r"task\s+(?:automatic\s+)?(\w+)\s*\("
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            tasks.append(match.group(1))
        return tasks

    def _extract_enums(self, content: str) -> List[str]:
        """
        Extract named enum typedef names from content.

        Pattern: typedef enum ... } NAME;

        Args:
            content: Source content to search

        Returns:
            List of enum type names
        """
        enums = []
        pattern = r"typedef\s+enum\s+.*?\}\s*(\w+)\s*;"
        matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            enums.append(match.group(1))
        return enums

    def _extract_structs(self, content: str) -> List[str]:
        """
        Extract struct typedef names from content.

        Pattern: typedef [packed] struct ... } NAME;

        Args:
            content: Source content to search

        Returns:
            List of struct type names
        """
        structs = []
        pattern = r"typedef\s+(?:packed\s+)?struct\s+.*?\}\s*(\w+)\s*;"
        matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            structs.append(match.group(1))
        return structs

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
