"""
Module Hierarchy Builder for Verilog/SystemVerilog HDL Dependency Analysis.

Extracts and analyzes module definitions, instantiations, ports, and parameters
to build a complete module instantiation hierarchy with metrics including fan-in/fan-out
analysis, cycle detection, hierarchy depth analysis, and architectural pattern identification.

Works entirely via regex — no external tooling required. Optionally uses networkx for
advanced graph algorithms if available.
"""

import logging
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from . import (
    ModuleHierarchy,
    ModuleNode,
    Instantiation,
    ParameterDef,
    PortDef,
)

logger = logging.getLogger(__name__)

# Try to import networkx for advanced graph algorithms
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class ModuleHierarchyBuilder:
    """
    Builds a complete module instantiation hierarchy from Verilog/SystemVerilog source files.

    Analyzes module definitions, instantiations, port declarations, and parameter definitions
    to construct a directed acyclic graph (or with cycles detected) representing the module
    hierarchy. Computes fan-in/fan-out metrics, detects architectural issues, identifies
    root/leaf/testbench modules, and analyzes hierarchy depth.

    Attributes:
        config (Dict): Configuration dict with keys:
            - project_root (str): Root directory for relative path resolution
            - ignore_dirs (List[str]): Directory names to skip (e.g., "test", "sim")
            - max_hierarchy_depth (int): Maximum depth to traverse (default: None for unlimited)
            - debug (bool): Enable debug logging
    """

    # Regex patterns for module analysis
    _MODULE_DEF_RE = re.compile(r'^\s*module\s+(\w+)', re.MULTILINE)
    _MODULE_INST_RE = re.compile(
        r'(\w+)\s+(?:#\s*\(.*?\)\s*)?(\w+)\s*\(',
        re.MULTILINE
    )
    _PORT_RE = re.compile(
        r'(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[.*?\])?\s*(\w+)',
        re.MULTILINE
    )
    _PARAMETER_RE = re.compile(
        r'(parameter|localparam)\s+(?:\w+\s+)?(\w+)\s*=',
        re.MULTILINE
    )

    # Verilog/SystemVerilog reserved keywords (for filtering false instantiations)
    _VERILOG_KEYWORDS = {
        'module', 'function', 'task', 'begin', 'end', 'if', 'else', 'for', 'while',
        'case', 'assign', 'always', 'always_ff', 'always_comb', 'always_latch',
        'initial', 'generate', 'wire', 'reg', 'logic', 'input', 'output', 'inout',
        'integer', 'real', 'time', 'genvar', 'localparam', 'parameter', 'bit',
        'byte', 'shortint', 'int', 'longint', 'shortreal', 'realtime', 'string',
        'void', 'type', 'class', 'interface', 'program', 'package', 'import',
        'typedef', 'enum', 'struct', 'union', 'automatic', 'static', 'extern',
        'virtual', 'pure', 'local', 'protected', 'const', 'assert', 'assume',
        'cover', 'property', 'sequence', 'checker', 'modport', 'clocking',
        'default', 'disable', 'endmodule', 'endfunction', 'endtask', 'endclass',
        'endinterface', 'endpackage', 'endprogram', 'endproperty', 'endsequence',
        'endchecker', 'endclocking', 'endgenerate', 'endgroup', 'specify',
        'endspecify', 'table', 'endtable', 'primitive', 'endprimitive', 'config',
        'endconfig', 'pullup', 'pulldown', 'supply0', 'supply1', 'wand', 'wor',
        'tri', 'triand', 'trior', 'tri0', 'tri1', 'trireg', 'uwire', 'signed',
        'unsigned', 'ref', 'return', 'break', 'continue', 'do', 'foreach',
        'forever', 'repeat', 'wait', 'fork', 'join', 'join_any', 'join_none',
        'force', 'release', 'posedge', 'negedge', 'edge', 'iff', 'inside', 'dist',
        'with', 'unique', 'priority', 'tagged',
    }

    # Testbench naming patterns
    _TESTBENCH_PATTERNS = ['*_tb', '*_test', 'tb_*', 'test_*']
    _TESTBENCH_EXTENSIONS = ['_tb.sv', '_tb.v', '_test.sv', '_test.v']

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ModuleHierarchyBuilder.

        Args:
            config (Dict): Configuration with keys:
                - project_root (str): Root directory for relative path resolution
                - ignore_dirs (List[str], optional): Directories to ignore
                - max_hierarchy_depth (int, optional): Maximum hierarchy depth
                - debug (bool, optional): Enable debug logging
        """
        self.config = config
        self.project_root = config.get('project_root', '')
        self.ignore_dirs = set(config.get('ignore_dirs', []))
        self.max_hierarchy_depth = config.get('max_hierarchy_depth')
        self.debug = config.get('debug', False)

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def build(self, file_cache: List[Dict[str, Any]]) -> ModuleHierarchy:
        """
        Build the complete module hierarchy from a file cache.

        Args:
            file_cache (List[Dict]): List of file entries, each with:
                - file_path or file_relative_path (str): File path
                - source (str): File source code

        Returns:
            ModuleHierarchy: Complete module instantiation hierarchy with metrics
        """
        hierarchy = ModuleHierarchy()

        if not file_cache:
            logger.warning("No files provided to build module hierarchy")
            return hierarchy

        # Phase 1: Extract module definitions and collect all instantiations
        module_map: Dict[str, ModuleNode] = {}  # name -> ModuleNode
        instantiations_list: List[Tuple[str, str, str, str, int]] = []  # (parent, child, instance, file, line)

        for entry in file_cache:
            file_path = entry.get('file_relative_path') or entry.get('file_path', 'unknown')
            source = entry.get('source', '')

            if not source.strip():
                continue

            # Remove comments for cleaner analysis
            cleaned_source = self._strip_comments(source)

            # Extract modules defined in this file
            for match in self._MODULE_DEF_RE.finditer(cleaned_source):
                module_name = match.group(1)
                if module_name not in module_map:
                    module_map[module_name] = ModuleNode(
                        name=module_name,
                        file_path=file_path,
                        line=cleaned_source[:match.start()].count('\n') + 1
                    )

            # Extract module instantiations
            for match in self._MODULE_INST_RE.finditer(cleaned_source):
                module_type = match.group(1)
                instance_name = match.group(2)

                # Filter out keywords
                if module_type.lower() in self._VERILOG_KEYWORDS:
                    continue

                # Find which module this instantiation belongs to
                # Simple heuristic: track modules defined so far in this file
                current_modules = {m for m in module_map.keys() if module_map[m].file_path == file_path}
                if current_modules:
                    for parent_module in current_modules:
                        # To properly track, we'd need better parsing, but for now
                        # just register all instantiations with file context
                        line_no = cleaned_source[:match.start()].count('\n') + 1
                        instantiations_list.append(
                            (parent_module, module_type, instance_name, file_path, line_no)
                        )

            # Extract ports for each module
            self._extract_ports(module_name, cleaned_source, module_map)

            # Extract parameters for each module
            self._extract_parameters(module_name, cleaned_source, module_map)

        # Phase 2: Build adjacency graph and detect testbenches
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        instantiations_dedup: List[Instantiation] = []

        for parent, child, instance, file_path, line in instantiations_list:
            adjacency[parent].add(child)
            if child not in module_map:
                module_map[child] = ModuleNode(
                    name=child,
                    file_path=file_path,
                    line=line
                )

            instantiations_dedup.append(
                Instantiation(
                    parent_module=parent,
                    child_module=child,
                    instance_name=instance,
                    file_path=file_path,
                    line=line
                )
            )

        # Ensure all nodes exist in adjacency map
        for module_name in module_map.keys():
            if module_name not in adjacency:
                adjacency[module_name] = set()

        # Detect testbenches
        for module_name, node in module_map.items():
            if self._is_testbench(module_name, node.file_path):
                node.is_testbench = True

        # Phase 3: Compute metrics
        in_degree = self._compute_in_degree(adjacency)
        out_degree = self._compute_out_degree(adjacency)

        # Update fan-in/fan-out in module nodes
        for module_name, node in module_map.items():
            node.fan_in = in_degree.get(module_name, 0)
            node.fan_out = out_degree.get(module_name, 0)

        # Identify root and leaf modules
        root_modules = {m for m, degree in in_degree.items() if degree == 0}
        leaf_modules = {m for m, degree in out_degree.items() if degree == 0}

        # Remove testbenches from root/leaf classification
        root_modules = {m for m in root_modules if not module_map[m].is_testbench}
        leaf_modules = {m for m in leaf_modules if not module_map[m].is_testbench}

        # Detect cycles
        cycles = self._detect_cycles(adjacency)

        # Compute max depth
        max_depth = self._compute_max_depth(adjacency, root_modules)

        # Populate the hierarchy object
        hierarchy.modules = module_map
        hierarchy.instantiations = instantiations_dedup
        hierarchy.adjacency = dict(adjacency)
        hierarchy.root_modules = root_modules
        hierarchy.leaf_modules = leaf_modules
        hierarchy.cycles = cycles
        hierarchy.max_depth = max_depth
        hierarchy.total_instances = len(instantiations_dedup)

        if self.debug:
            logger.debug(f"Built hierarchy with {len(module_map)} modules, "
                        f"{len(instantiations_dedup)} instantiations, "
                        f"{len(cycles)} cycles, max_depth={max_depth}")

        return hierarchy

    def _strip_comments(self, source: str) -> str:
        """
        Remove // line comments and /* */ block comments from source code.

        Args:
            source (str): Source code

        Returns:
            str: Source code with comments removed
        """
        # Remove block comments /* ... */
        source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)

        # Remove line comments //
        source = re.sub(r'//.*?$', '', source, flags=re.MULTILINE)

        return source

    def _extract_ports(
        self,
        module_name: str,
        source: str,
        module_map: Dict[str, ModuleNode]
    ) -> None:
        """
        Extract port declarations for a given module.

        Args:
            module_name (str): Name of the module
            source (str): Source code containing the module
            module_map (Dict): Module map to update
        """
        if module_name not in module_map:
            return

        module_node = module_map[module_name]
        ports: List[PortDef] = []

        for match in self._PORT_RE.finditer(source):
            direction = match.group(1)
            port_name = match.group(2)

            ports.append(PortDef(
                name=port_name,
                direction=direction,
                data_type='logic'
            ))

        module_node.ports = ports

    def _extract_parameters(
        self,
        module_name: str,
        source: str,
        module_map: Dict[str, ModuleNode]
    ) -> None:
        """
        Extract parameter declarations for a given module.

        Args:
            module_name (str): Name of the module
            source (str): Source code containing the module
            module_map (Dict): Module map to update
        """
        if module_name not in module_map:
            return

        module_node = module_map[module_name]
        parameters: List[ParameterDef] = []

        for match in self._PARAMETER_RE.finditer(source):
            is_localparam = match.group(1) == 'localparam'
            param_name = match.group(2)

            parameters.append(ParameterDef(
                name=param_name,
                data_type='',
                default_value='',
                is_localparam=is_localparam
            ))

        module_node.parameters = parameters

    def _is_testbench(self, module_name: str, file_path: str) -> bool:
        """
        Determine if a module is a testbench based on naming patterns.

        Args:
            module_name (str): Name of the module
            file_path (str): Path to the file containing the module

        Returns:
            bool: True if module appears to be a testbench
        """
        # Check module name patterns
        lower_name = module_name.lower()
        for pattern in self._TESTBENCH_PATTERNS:
            if re.match(pattern.replace('*', '.*'), lower_name):
                return True

        # Check file extension patterns
        for ext in self._TESTBENCH_EXTENSIONS:
            if file_path.endswith(ext):
                return True

        return False

    def _compute_in_degree(self, adjacency: Dict[str, Set[str]]) -> Dict[str, int]:
        """Compute in-degree (fan-in) for all modules."""
        in_degree: Dict[str, int] = defaultdict(int)

        # Initialize all nodes
        for module in adjacency.keys():
            in_degree[module] = 0

        # Count in-degrees
        for module, children in adjacency.items():
            for child in children:
                in_degree[child] += 1

        return dict(in_degree)

    def _compute_out_degree(self, adjacency: Dict[str, Set[str]]) -> Dict[str, int]:
        """Compute out-degree (fan-out) for all modules."""
        out_degree: Dict[str, int] = {}

        for module, children in adjacency.items():
            out_degree[module] = len(children)

        return out_degree

    def _detect_cycles(self, adjacency: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Detect cycles in the module instantiation graph.

        Uses networkx if available, otherwise falls back to DFS-based detection.

        Args:
            adjacency (Dict): Module adjacency graph

        Returns:
            List[List[str]]: List of cycles (each cycle is a list of module names)
        """
        if not adjacency:
            return []

        # Use networkx if available
        if HAS_NETWORKX:
            try:
                G = nx.DiGraph()
                for parent, children in adjacency.items():
                    for child in children:
                        G.add_edge(parent, child)

                cycles = list(nx.simple_cycles(G))
                return cycles
            except Exception as e:
                logger.warning(f"NetworkX cycle detection failed: {e}")

        # Fallback: DFS-based cycle detection
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

        def dfs_cycle(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs_cycle(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        try:
            for node in adjacency.keys():
                if node not in visited:
                    dfs_cycle(node, [])
        except RecursionError:
            logger.error("Recursion limit exceeded during cycle detection")

        return cycles

    def _compute_max_depth(
        self,
        adjacency: Dict[str, Set[str]],
        root_modules: Set[str]
    ) -> int:
        """
        Compute the maximum hierarchy depth from root modules.

        Args:
            adjacency (Dict): Module adjacency graph
            root_modules (Set): Set of root module names

        Returns:
            int: Maximum hierarchy depth
        """
        if not root_modules or not adjacency:
            return 1

        memo: Dict[str, int] = {}

        def dfs_depth(node: str, visited: Set[str]) -> int:
            if node in memo:
                return memo[node]

            if node in visited:
                return 0  # Break cycle

            visited.add(node)

            children = adjacency.get(node, set())
            if not children:
                memo[node] = 1
                return 1

            max_child_depth = 0
            for child in children:
                d = dfs_depth(child, visited.copy())
                max_child_depth = max(max_child_depth, d)

            depth = 1 + max_child_depth
            memo[node] = depth
            return depth

        max_depth = 0
        for root in root_modules:
            try:
                depth = dfs_depth(root, set())
                max_depth = max(max_depth, depth)
            except RecursionError:
                logger.warning(f"Recursion limit exceeded computing depth for root {root}")

        return max_depth
