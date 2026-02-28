"""
Include Dependency Graph Builder for Verilog/SystemVerilog HDL Analysis.

Resolves `include directives (both "local" and <system> styles), builds a complete
include dependency tree tracking transitive includes, detects circular include chains,
and computes include depth metrics.

Works entirely via regex — no external tooling required.
"""

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import IncludeTree, ResolvedInclude

logger = logging.getLogger(__name__)


class IncludeDependencyGraph:
    """
    Builds a complete include dependency tree for Verilog/SystemVerilog projects.

    Analyzes `include directives in source files, resolves them against the project
    directory structure and configured include paths, tracks transitive includes,
    detects circular dependencies, and computes metrics like maximum include depth
    and unresolved include count.

    Attributes:
        config (Dict): Configuration dict with keys:
            - project_root (str): Root directory for relative path resolution
            - include_paths (List[str]): Additional include search paths
            - max_include_depth (int): Maximum include nesting depth to follow
            - debug (bool): Enable debug logging
    """

    # Regex patterns for include directives
    # Matches: `include "path/to/file.vh"
    _INCLUDE_QUOTED_RE = re.compile(
        r'^\s*`\s*include\s+"([^"]+)"',
        re.MULTILINE
    )

    # Matches: `include <system/header.h>
    _INCLUDE_SYSTEM_RE = re.compile(
        r'^\s*`\s*include\s+<([^>]+)>',
        re.MULTILINE
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the IncludeDependencyGraph builder.

        Args:
            config (Dict): Configuration with keys:
                - project_root (str): Root directory for relative path resolution
                - include_paths (List[str], optional): Additional include search paths
                - max_include_depth (int, optional): Maximum include depth (default: 20)
                - debug (bool, optional): Enable debug logging
        """
        self.config = config
        self.project_root = Path(config.get('project_root', '.'))
        self.include_paths = [
            Path(p) if not Path(p).is_absolute() else Path(p)
            for p in config.get('include_paths', [])
        ]
        self.max_include_depth = config.get('max_include_depth', 20)
        self.debug = config.get('debug', False)

        if self.debug:
            logger.setLevel(logging.DEBUG)

        # Cache for resolved files to avoid infinite recursion
        self._file_cache: Dict[str, Set[str]] = {}  # file -> set of includes it directly includes

    def build(self, file_cache: List[Dict[str, Any]]) -> IncludeTree:
        """
        Build the complete include dependency tree from a file cache.

        Args:
            file_cache (List[Dict]): List of file entries, each with:
                - file_path or file_relative_path (str): File path
                - source (str): File source code

        Returns:
            IncludeTree: Complete include dependency analysis results
        """
        tree = IncludeTree()

        if not file_cache:
            logger.warning("No files provided to build include dependency tree")
            return tree

        # Phase 1: Extract all direct include directives from all files
        direct_includes: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        # Maps: source_file -> [(include_name, include_type, line_number), ...]

        for entry in file_cache:
            file_path = entry.get('file_relative_path') or entry.get('file_path', 'unknown')
            source = entry.get('source', '')

            if not source.strip():
                continue

            # Find all quoted includes
            for match in self._INCLUDE_QUOTED_RE.finditer(source):
                include_name = match.group(1)
                line_no = source[:match.start()].count('\n') + 1
                direct_includes[file_path].append((include_name, 'local', line_no))

            # Find all system includes
            for match in self._INCLUDE_SYSTEM_RE.finditer(source):
                include_name = match.group(1)
                line_no = source[:match.start()].count('\n') + 1
                direct_includes[file_path].append((include_name, 'system', line_no))

        # Phase 2: Resolve all includes and build transitive closure
        includes_by_file: Dict[str, List[ResolvedInclude]] = defaultdict(list)
        unresolved_includes: List[ResolvedInclude] = []
        circular_includes: List[Tuple[str, str]] = []
        max_depth = 0

        for source_file, includes in direct_includes.items():
            for include_name, include_type, line_no in includes:
                resolved_include = ResolvedInclude(
                    include_name=include_name,
                    source_file=source_file,
                    include_depth=0,
                    is_system=(include_type == 'system'),
                    line=line_no
                )

                # Resolve the include
                resolved_path = self._resolve_include_path(
                    include_name,
                    source_file,
                    include_type
                )

                if resolved_path:
                    resolved_include.resolved_path = resolved_path
                    resolved_include.resolved = True
                    includes_by_file[source_file].append(resolved_include)

                    # Recursively follow includes up to max_include_depth
                    transitive = self._follow_includes(
                        resolved_path,
                        source_file,
                        depth=1,
                        visited_chain=set([source_file, resolved_path]),
                        circular_includes=circular_includes
                    )

                    # Update max depth
                    max_depth = max(max_depth, len(transitive) + 1)

                    # Record transitive includes
                    for trans_path in transitive:
                        trans_include = ResolvedInclude(
                            include_name=os.path.basename(trans_path),
                            source_file=source_file,
                            resolved_path=trans_path,
                            include_depth=1 + len(transitive),
                            resolved=True,
                            is_system=False,
                            line=0
                        )
                        includes_by_file[source_file].append(trans_include)
                else:
                    unresolved_includes.append(resolved_include)
                    if self.debug:
                        logger.debug(f"Unresolved include: {include_name} from {source_file}")

        # Phase 3: Build transitive closure (file -> all files it includes)
        include_chains: Dict[str, List[str]] = {}
        for source_file, resolved_list in includes_by_file.items():
            chains = set()
            for resolved in resolved_list:
                if resolved.resolved_path:
                    chains.add(resolved.resolved_path)
            include_chains[source_file] = sorted(list(chains))

        # Populate tree
        tree.includes_by_file = dict(includes_by_file)
        tree.include_chains = include_chains
        tree.circular_includes = circular_includes
        tree.unresolved_includes = unresolved_includes
        tree.max_depth = max_depth
        tree.total_includes = sum(len(lst) for lst in includes_by_file.values())

        if self.debug:
            logger.debug(f"Built include tree with {len(includes_by_file)} files, "
                        f"{tree.total_includes} total includes, "
                        f"{len(unresolved_includes)} unresolved, "
                        f"{len(circular_includes)} circular, max_depth={max_depth}")

        return tree

    def _resolve_include_path(
        self,
        include_name: str,
        source_file: str,
        include_type: str
    ) -> Optional[str]:
        """
        Resolve an include directive to an actual file path.

        Uses this search order:
        1. Relative to the directory of the source file (for local includes)
        2. Relative to project_root
        3. Each path in config.include_paths (relative to project_root)

        Args:
            include_name (str): Include name as written in source (e.g., "subdir/header.vh")
            source_file (str): Path to the source file containing the include
            include_type (str): "local" or "system"

        Returns:
            Optional[str]: Absolute path to the resolved include file, or None if unresolved
        """
        search_dirs: List[Path] = []

        # For local includes, search relative to source file directory first
        if include_type == 'local':
            source_dir = self.project_root / source_file
            if not source_dir.is_absolute():
                source_dir = (self.project_root / source_file).resolve()
            source_dir = source_dir.parent
            search_dirs.append(source_dir)

        # Add project root
        search_dirs.append(self.project_root)

        # Add configured include paths
        for inc_path in self.include_paths:
            if not inc_path.is_absolute():
                search_dirs.append(self.project_root / inc_path)
            else:
                search_dirs.append(inc_path)

        # Try to find the file in search directories
        for search_dir in search_dirs:
            candidate = search_dir / include_name

            if candidate.is_file():
                try:
                    return str(candidate.resolve())
                except Exception as e:
                    logger.warning(f"Error resolving path {candidate}: {e}")
                    continue

        # Fallback: try relative to project root alone
        candidate = self.project_root / include_name
        if candidate.is_file():
            try:
                return str(candidate.resolve())
            except Exception as e:
                logger.warning(f"Error resolving path {candidate}: {e}")

        return None

    def _follow_includes(
        self,
        file_path: str,
        source_file: str,
        depth: int,
        visited_chain: Set[str],
        circular_includes: List[Tuple[str, str]]
    ) -> List[str]:
        """
        Recursively follow includes in a file up to max_include_depth.

        Args:
            file_path (str): Absolute path to the file to scan for includes
            source_file (str): Original source file (for circular detection)
            depth (int): Current recursion depth
            visited_chain (Set[str]): Set of files visited in this chain (for circular detection)
            circular_includes (List): List to append circular include pairs to

        Returns:
            List[str]: Transitive list of all included files
        """
        transitive: List[str] = []

        # Check depth limit
        if depth >= self.max_include_depth:
            return transitive

        # Check if file is already in visited chain (circular)
        if file_path in visited_chain:
            circular_includes.append((source_file, file_path))
            return transitive

        # Check cache
        if file_path in self._file_cache:
            # We've already processed this file
            return list(self._file_cache[file_path])

        # Read and parse file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Cannot read file {file_path}: {e}")
            return transitive

        # Extract includes from this file
        direct_includes: List[Tuple[str, str]] = []

        for match in self._INCLUDE_QUOTED_RE.finditer(content):
            include_name = match.group(1)
            direct_includes.append((include_name, 'local'))

        for match in self._INCLUDE_SYSTEM_RE.finditer(content):
            include_name = match.group(1)
            direct_includes.append((include_name, 'system'))

        # Resolve and recursively follow each include
        new_visited = visited_chain.copy()
        new_visited.add(file_path)

        for include_name, include_type in direct_includes:
            resolved_path = self._resolve_include_path(
                include_name,
                file_path,
                include_type
            )

            if resolved_path and resolved_path not in new_visited:
                transitive.append(resolved_path)

                # Recursively follow
                sub_transitive = self._follow_includes(
                    resolved_path,
                    source_file,
                    depth + 1,
                    new_visited,
                    circular_includes
                )
                transitive.extend(sub_transitive)

        # Cache this result
        self._file_cache[file_path] = set(transitive)

        return transitive
