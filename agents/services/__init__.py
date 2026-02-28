"""
HDL Dependency Analysis — Data Models.

Canonical dataclass definitions for all dependency graph structures used across
the CARE framework's HDL dependency analysis pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Primitive / Shared Types
# ---------------------------------------------------------------------------

@dataclass
class PortDef:
    """A single port declaration in a module or interface."""
    name: str
    direction: str  # input | output | inout | ref
    data_type: str = "logic"
    width: str = ""  # e.g. "[7:0]"
    file_path: str = ""
    line: int = 0


@dataclass
class ParameterDef:
    """A parameter or localparam declaration."""
    name: str
    data_type: str = ""
    default_value: str = ""
    is_localparam: bool = False
    file_path: str = ""
    line: int = 0


@dataclass
class SymbolDef:
    """A resolved symbol (signal, type, macro, function, task)."""
    name: str
    kind: str  # signal | type | parameter | macro | function | task | port | enum | struct
    data_type: str = ""
    file_path: str = ""
    line: int = 0
    scope: str = ""  # module::name, package::name, or global
    value: str = ""  # for macros/parameters


# ---------------------------------------------------------------------------
# Module Hierarchy
# ---------------------------------------------------------------------------

@dataclass
class ModuleNode:
    """Metadata for a single module definition."""
    name: str
    file_path: str
    line: int = 0
    ports: List[PortDef] = field(default_factory=list)
    parameters: List[ParameterDef] = field(default_factory=list)
    fan_in: int = 0
    fan_out: int = 0
    is_testbench: bool = False
    is_library: bool = False


@dataclass
class Instantiation:
    """A single module instantiation record."""
    parent_module: str
    child_module: str
    instance_name: str
    file_path: str
    line: int = 0
    parameter_overrides: Dict[str, str] = field(default_factory=dict)
    port_connections: Dict[str, str] = field(default_factory=dict)
    inside_generate: bool = False
    generate_condition: str = ""


@dataclass
class ModuleHierarchy:
    """Complete module instantiation hierarchy."""
    modules: Dict[str, ModuleNode] = field(default_factory=dict)
    instantiations: List[Instantiation] = field(default_factory=list)
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)  # parent -> {children}
    root_modules: Set[str] = field(default_factory=set)
    leaf_modules: Set[str] = field(default_factory=set)
    cycles: List[List[str]] = field(default_factory=list)
    max_depth: int = 0
    total_instances: int = 0

    def get_children(self, module_name: str) -> Set[str]:
        return self.adjacency.get(module_name, set())

    def get_parents(self, module_name: str) -> Set[str]:
        parents = set()
        for parent, children in self.adjacency.items():
            if module_name in children:
                parents.add(parent)
        return parents


# ---------------------------------------------------------------------------
# Include Dependency Tree
# ---------------------------------------------------------------------------

@dataclass
class ResolvedInclude:
    """A single resolved `include directive."""
    include_name: str  # as written in source
    source_file: str  # which file contains the `include
    resolved_path: Optional[str] = None
    include_depth: int = 0
    resolved: bool = False
    is_system: bool = False
    line: int = 0


@dataclass
class IncludeTree:
    """Complete include dependency tree."""
    includes_by_file: Dict[str, List[ResolvedInclude]] = field(default_factory=dict)
    include_chains: Dict[str, List[str]] = field(default_factory=dict)  # file -> transitive includes
    circular_includes: List[Tuple[str, str]] = field(default_factory=list)
    unresolved_includes: List[ResolvedInclude] = field(default_factory=list)
    max_depth: int = 0
    total_includes: int = 0


# ---------------------------------------------------------------------------
# Package Import Map
# ---------------------------------------------------------------------------

@dataclass
class PackageDefinition:
    """A SystemVerilog package definition."""
    name: str
    file_path: str
    line: int = 0
    exported_symbols: Set[str] = field(default_factory=set)
    parameters: List[ParameterDef] = field(default_factory=list)
    typedefs: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)


@dataclass
class ImportStatement:
    """A single import statement."""
    package_name: str
    symbol: str  # specific symbol or "*" for wildcard
    file_path: str
    line: int = 0
    module_scope: str = ""  # which module contains this import


@dataclass
class PackageImportMap:
    """Complete package import resolution."""
    imports_by_file: Dict[str, List[ImportStatement]] = field(default_factory=dict)
    package_defs: Dict[str, PackageDefinition] = field(default_factory=dict)
    symbol_to_files: Dict[str, Set[str]] = field(default_factory=dict)  # symbol -> files importing it
    unresolved_packages: Set[str] = field(default_factory=set)
    total_imports: int = 0


# ---------------------------------------------------------------------------
# Parameter Propagation
# ---------------------------------------------------------------------------

@dataclass
class ParameterOverride:
    """A parameter override in a module instantiation."""
    instance_name: str
    parent_module: str
    child_module: str
    param_name: str
    override_value: str
    default_value: str = ""
    param_type: str = ""
    file_path: str = ""
    line: int = 0
    type_mismatch: bool = False
    mismatch_detail: str = ""


@dataclass
class ParameterPropagationMap:
    """Complete parameter propagation tracking."""
    overrides: List[ParameterOverride] = field(default_factory=list)
    overrides_by_instance: Dict[str, List[ParameterOverride]] = field(default_factory=dict)
    type_mismatches: List[ParameterOverride] = field(default_factory=list)
    total_overrides: int = 0
    total_mismatches: int = 0


# ---------------------------------------------------------------------------
# Interface / Modport Bindings
# ---------------------------------------------------------------------------

@dataclass
class ModportDef:
    """A modport declaration within an interface."""
    name: str
    interface_name: str
    ports: List[PortDef] = field(default_factory=list)
    file_path: str = ""
    line: int = 0


@dataclass
class InterfaceDefinition:
    """A SystemVerilog interface definition."""
    name: str
    file_path: str
    line: int = 0
    modports: List[ModportDef] = field(default_factory=list)
    parameters: List[ParameterDef] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)


@dataclass
class InterfaceBinding:
    """An interface instantiation / binding in a module."""
    interface_name: str
    instance_name: str
    modport_name: str = ""
    parent_module: str = ""
    file_path: str = ""
    line: int = 0


@dataclass
class InterfaceBindingMap:
    """Complete interface binding analysis."""
    interface_defs: Dict[str, InterfaceDefinition] = field(default_factory=dict)
    bindings: List[InterfaceBinding] = field(default_factory=list)
    modport_usage: Dict[str, List[str]] = field(default_factory=dict)  # modport -> [modules]
    unconnected_interfaces: List[str] = field(default_factory=list)
    total_interfaces: int = 0
    total_bindings: int = 0


# ---------------------------------------------------------------------------
# Generate Block Expansion
# ---------------------------------------------------------------------------

@dataclass
class GenerateBlock:
    """A generate block with conditional instantiation."""
    block_type: str  # "if" | "for" | "case"
    condition: str  # the generate condition expression
    parent_module: str
    file_path: str
    line: int = 0
    contained_instances: List[str] = field(default_factory=list)  # module names instantiated
    label: str = ""  # generate block label if present


@dataclass
class GenerateBlockExpansions:
    """Complete generate block tracking."""
    blocks: List[GenerateBlock] = field(default_factory=list)
    blocks_by_module: Dict[str, List[GenerateBlock]] = field(default_factory=dict)
    conditional_instances: Dict[str, str] = field(default_factory=dict)  # instance -> condition
    total_generate_blocks: int = 0
    total_conditional_instances: int = 0


# ---------------------------------------------------------------------------
# Symbol Table
# ---------------------------------------------------------------------------

@dataclass
class SymbolTable:
    """Cross-file symbol resolution table."""
    symbols: Dict[str, SymbolDef] = field(default_factory=dict)  # qualified_name -> def
    signals: Dict[str, SymbolDef] = field(default_factory=dict)
    types: Dict[str, SymbolDef] = field(default_factory=dict)
    parameters: Dict[str, SymbolDef] = field(default_factory=dict)
    macros: Dict[str, SymbolDef] = field(default_factory=dict)
    functions: Dict[str, SymbolDef] = field(default_factory=dict)
    tasks: Dict[str, SymbolDef] = field(default_factory=dict)
    collisions: List[Tuple[str, List[SymbolDef]]] = field(default_factory=list)
    total_symbols: int = 0
    total_collisions: int = 0

    def lookup(self, name: str, scope: str = "") -> Optional[SymbolDef]:
        """Look up a symbol, optionally scoped."""
        if scope:
            qualified = f"{scope}::{name}"
            if qualified in self.symbols:
                return self.symbols[qualified]
        return self.symbols.get(name)

    def add(self, sym: SymbolDef) -> None:
        """Add a symbol to the table."""
        qualified = f"{sym.scope}::{sym.name}" if sym.scope else sym.name
        # Track in category-specific dict
        category_map = {
            "signal": self.signals,
            "port": self.signals,
            "type": self.types,
            "enum": self.types,
            "struct": self.types,
            "parameter": self.parameters,
            "macro": self.macros,
            "function": self.functions,
            "task": self.tasks,
        }
        target = category_map.get(sym.kind, self.symbols)
        if qualified in self.symbols:
            # Collision detection
            existing = self.symbols[qualified]
            if existing.file_path != sym.file_path:
                self.collisions.append((qualified, [existing, sym]))
                self.total_collisions += 1
        self.symbols[qualified] = sym
        target[qualified] = sym
        self.total_symbols += 1


# ---------------------------------------------------------------------------
# Top-Level Dependency Graph
# ---------------------------------------------------------------------------

@dataclass
class AnalysisMetadata:
    """Metadata about the dependency analysis run."""
    project_root: str = ""
    files_analyzed: int = 0
    analysis_time_seconds: float = 0.0
    verible_available: bool = False
    verible_used: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DependencyGraph:
    """Top-level container for all dependency analysis results."""
    module_hierarchy: ModuleHierarchy = field(default_factory=ModuleHierarchy)
    include_tree: IncludeTree = field(default_factory=IncludeTree)
    package_imports: PackageImportMap = field(default_factory=PackageImportMap)
    parameter_map: ParameterPropagationMap = field(default_factory=ParameterPropagationMap)
    interface_bindings: InterfaceBindingMap = field(default_factory=InterfaceBindingMap)
    generate_expansions: GenerateBlockExpansions = field(default_factory=GenerateBlockExpansions)
    symbol_table: SymbolTable = field(default_factory=SymbolTable)
    metadata: AnalysisMetadata = field(default_factory=AnalysisMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON export / backward compat."""
        from dataclasses import asdict
        return asdict(self)

    @property
    def total_modules(self) -> int:
        return len(self.module_hierarchy.modules)

    @property
    def total_includes(self) -> int:
        return self.include_tree.total_includes

    @property
    def total_packages(self) -> int:
        return len(self.package_imports.package_defs)

    @property
    def total_interfaces(self) -> int:
        return self.interface_bindings.total_interfaces

    @property
    def has_cycles(self) -> bool:
        return len(self.module_hierarchy.cycles) > 0

    @property
    def has_unresolved_includes(self) -> bool:
        return len(self.include_tree.unresolved_includes) > 0

    @property
    def score_summary(self) -> Dict[str, Any]:
        """Quick summary for scoring."""
        return {
            "modules": self.total_modules,
            "instantiations": self.module_hierarchy.total_instances,
            "max_depth": self.module_hierarchy.max_depth,
            "cycles": len(self.module_hierarchy.cycles),
            "includes": self.total_includes,
            "unresolved_includes": len(self.include_tree.unresolved_includes),
            "circular_includes": len(self.include_tree.circular_includes),
            "packages": self.total_packages,
            "imports": self.package_imports.total_imports,
            "unresolved_packages": len(self.package_imports.unresolved_packages),
            "parameter_overrides": self.parameter_map.total_overrides,
            "param_mismatches": self.parameter_map.total_mismatches,
            "interfaces": self.total_interfaces,
            "interface_bindings": self.interface_bindings.total_bindings,
            "generate_blocks": self.generate_expansions.total_generate_blocks,
            "symbols": self.symbol_table.total_symbols,
            "symbol_collisions": self.symbol_table.total_collisions,
        }
