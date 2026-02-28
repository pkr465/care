"""
HDL Dependency Graph Analyzer — Main Orchestrator.

Coordinates all 7 dependency analysis services and provides backward-compatible
APIs for graph building, documentation, validation, and modularization analysis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Import services with graceful degradation
try:
    from agents.services.module_hierarchy_builder import ModuleHierarchyBuilder
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import ModuleHierarchyBuilder")
    ModuleHierarchyBuilder = None

try:
    from agents.services.include_dependency_graph import IncludeDependencyGraph
except ImportError:
    IncludeDependencyGraph = None

try:
    from agents.services.package_import_resolver import PackageImportResolver
except ImportError:
    PackageImportResolver = None

try:
    from agents.services.parameter_propagation_tracker import ParameterPropagationTracker
except ImportError:
    ParameterPropagationTracker = None

try:
    from agents.services.interface_binding_analyzer import InterfaceBindingAnalyzer
except ImportError:
    InterfaceBindingAnalyzer = None

try:
    from agents.services.generate_block_expander import GenerateBlockExpander
except ImportError:
    GenerateBlockExpander = None

try:
    from agents.services.symbol_table_builder import SymbolTableBuilder
except ImportError:
    SymbolTableBuilder = None

# Import data models
try:
    from agents.services import (
        DependencyGraph,
        AnalysisMetadata,
        ModuleHierarchy,
        IncludeTree,
        PackageImportMap,
        ParameterPropagationMap,
        InterfaceBindingMap,
        GenerateBlockExpansions,
        SymbolTable,
    )
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Could not import data models from agents.services")
    raise

# Import Verible wrapper
try:
    from agents.core.verible_parser_wrapper import VeribleParserWrapper
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import VeribleParserWrapper; Verible support disabled")
    VeribleParserWrapper = None

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for HDL dependency analyzer."""
    project_root: str
    ignore_dirs: List[str] = field(default_factory=lambda: [".git", "build", "dist"])
    include_paths: List[str] = field(default_factory=list)
    max_include_depth: int = 2
    max_hierarchy_depth: int = 10
    use_verible: bool = True
    verible_timeout: int = 30
    exclude_system_packages: bool = True
    track_parameters: bool = True
    track_interfaces: bool = True
    debug: bool = False


class HDLDependencyAnalyzer:
    """
    Main orchestrator for HDL dependency analysis.

    Coordinates all 7 services:
    1. ModuleHierarchyBuilder — module definitions and instantiations
    2. IncludeDependencyGraph — `include resolution
    3. PackageImportResolver — package definitions and imports
    4. ParameterPropagationTracker — parameter overrides and type mismatches
    5. InterfaceBindingAnalyzer — interface definitions and bindings
    6. GenerateBlockExpander — generate block conditional logic
    7. SymbolTableBuilder — cross-file symbol resolution

    Provides backward-compatible APIs:
    - build_graph() — returns dict for codebase_static_agent.py
    - document_graph() — generates documentation with summary stats
    - propose_modularization() — suggests improvements
    - validate_modularization() — checks for cycles, depth, orphans
    - get_module_dependencies() — single-file dependency query
    """

    def __init__(self, config: AnalyzerConfig):
        """Initialize analyzer with configuration and service instances."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Build config dict for services
        svc_cfg = {
            "project_root": config.project_root,
            "ignore_dirs": config.ignore_dirs,
            "include_paths": config.include_paths,
            "max_include_depth": config.max_include_depth,
            "max_hierarchy_depth": config.max_hierarchy_depth,
            "exclude_system_packages": config.exclude_system_packages,
            "debug": config.debug,
        }

        # Instantiate all services with config
        self.hierarchy_builder = ModuleHierarchyBuilder(svc_cfg) if ModuleHierarchyBuilder else None
        self.include_graph = IncludeDependencyGraph(svc_cfg) if IncludeDependencyGraph else None
        self.package_resolver = PackageImportResolver(svc_cfg) if PackageImportResolver else None
        self.param_tracker = ParameterPropagationTracker(svc_cfg) if ParameterPropagationTracker else None
        self.interface_analyzer = InterfaceBindingAnalyzer(svc_cfg) if InterfaceBindingAnalyzer else None
        self.generate_expander = GenerateBlockExpander(svc_cfg) if GenerateBlockExpander else None
        self.symbol_builder = SymbolTableBuilder(svc_cfg) if SymbolTableBuilder else None

        # Optional Verible parser
        self.verible = None
        if config.use_verible and VeribleParserWrapper:
            try:
                self.verible = VeribleParserWrapper(timeout=config.verible_timeout)
            except Exception as e:
                self.logger.warning("Failed to initialize Verible: %s", e)

        self.logger.info(
            "HDLDependencyAnalyzer initialized with config: project_root=%s, debug=%s",
            config.project_root, config.debug
        )

    def analyze(self, file_cache: List[Dict]) -> DependencyGraph:
        """
        Main analysis orchestrator.

        Args:
            file_cache: List of {path, content, ...} dicts representing files

        Returns:
            DependencyGraph containing all analysis results
        """
        start_time = time.time()
        graph = DependencyGraph()
        graph.metadata.project_root = self.config.project_root
        graph.metadata.files_analyzed = len(file_cache)
        graph.metadata.verible_available = self.verible.available if self.verible else False

        try:
            # Phase 1: Module Hierarchy
            try:
                if self.hierarchy_builder:
                    self.logger.info("Building module hierarchy...")
                    graph.module_hierarchy = self.hierarchy_builder.build(file_cache)
                    self.logger.info(
                        "Module hierarchy complete: %d modules, %d instantiations",
                        len(graph.module_hierarchy.modules),
                        graph.module_hierarchy.total_instances
                    )
                else:
                    self.logger.warning("ModuleHierarchyBuilder not available; skipping")
            except Exception as e:
                error_msg = f"ModuleHierarchyBuilder failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

            # Phase 2: Include Dependency Graph
            try:
                if self.include_graph:
                    self.logger.info("Building include dependency graph...")
                    graph.include_tree = self.include_graph.build(file_cache)
                    self.logger.info(
                        "Include graph complete: %d total includes, max_depth=%d",
                        graph.include_tree.total_includes,
                        graph.include_tree.max_depth
                    )
                else:
                    self.logger.warning("IncludeDependencyGraph not available; skipping")
            except Exception as e:
                error_msg = f"IncludeDependencyGraph failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

            # Phase 3: Package Import Resolution
            try:
                if self.package_resolver:
                    self.logger.info("Resolving package imports...")
                    graph.package_imports = self.package_resolver.build(file_cache)
                    self.logger.info(
                        "Package imports resolved: %d packages, %d imports, %d unresolved",
                        len(graph.package_imports.package_defs),
                        graph.package_imports.total_imports,
                        len(graph.package_imports.unresolved_packages)
                    )
                else:
                    self.logger.warning("PackageImportResolver not available; skipping")
            except Exception as e:
                error_msg = f"PackageImportResolver failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

            # Phase 4: Parameter Propagation Tracking
            try:
                if self.param_tracker and self.config.track_parameters:
                    self.logger.info("Tracking parameter propagation...")
                    graph.parameter_map = self.param_tracker.build(file_cache, graph.module_hierarchy)
                    self.logger.info(
                        "Parameter tracking complete: %d overrides, %d mismatches",
                        graph.parameter_map.total_overrides,
                        graph.parameter_map.total_mismatches
                    )
                else:
                    if not self.config.track_parameters:
                        self.logger.info("Parameter tracking disabled by config")
                    else:
                        self.logger.warning("ParameterPropagationTracker not available; skipping")
            except Exception as e:
                error_msg = f"ParameterPropagationTracker failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

            # Phase 5: Interface Binding Analysis
            try:
                if self.interface_analyzer and self.config.track_interfaces:
                    self.logger.info("Analyzing interface bindings...")
                    graph.interface_bindings = self.interface_analyzer.build(file_cache)
                    self.logger.info(
                        "Interface analysis complete: %d interfaces, %d bindings",
                        graph.interface_bindings.total_interfaces,
                        graph.interface_bindings.total_bindings
                    )
                else:
                    if not self.config.track_interfaces:
                        self.logger.info("Interface tracking disabled by config")
                    else:
                        self.logger.warning("InterfaceBindingAnalyzer not available; skipping")
            except Exception as e:
                error_msg = f"InterfaceBindingAnalyzer failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

            # Phase 6: Generate Block Expansion
            try:
                if self.generate_expander:
                    self.logger.info("Expanding generate blocks...")
                    graph.generate_expansions = self.generate_expander.build(file_cache, graph.module_hierarchy)
                    self.logger.info(
                        "Generate block expansion complete: %d blocks, %d conditional instances",
                        graph.generate_expansions.total_generate_blocks,
                        graph.generate_expansions.total_conditional_instances
                    )
                else:
                    self.logger.warning("GenerateBlockExpander not available; skipping")
            except Exception as e:
                error_msg = f"GenerateBlockExpander failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

            # Phase 7: Symbol Table Building
            try:
                if self.symbol_builder:
                    self.logger.info("Building symbol table...")
                    graph.symbol_table = self.symbol_builder.build(
                        file_cache,
                        graph.include_tree,
                        graph.package_imports
                    )
                    self.logger.info(
                        "Symbol table complete: %d symbols, %d collisions",
                        graph.symbol_table.total_symbols,
                        graph.symbol_table.total_collisions
                    )
                else:
                    self.logger.warning("SymbolTableBuilder not available; skipping")
            except Exception as e:
                error_msg = f"SymbolTableBuilder failed: {str(e)[:100]}"
                self.logger.error(error_msg)
                graph.metadata.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error in analysis pipeline: {str(e)}"
            self.logger.error(error_msg)
            graph.metadata.errors.append(error_msg)

        # Update metadata
        graph.metadata.analysis_time_seconds = time.time() - start_time
        graph.metadata.verible_used = bool(self.verible and self.verible.available)

        self.logger.info(
            "Analysis complete in %.2f seconds with %d errors",
            graph.metadata.analysis_time_seconds,
            len(graph.metadata.errors)
        )

        return graph

    def build_graph(self, file_cache: List[Dict]) -> Dict:
        """
        Backward-compatible API: build graph and return as dict.

        Called by codebase_static_agent.py.

        Args:
            file_cache: List of {path, content, ...} dicts

        Returns:
            Dictionary representation of DependencyGraph
        """
        graph = self.analyze(file_cache)
        return graph.to_dict()

    def document_graph(self, graph: Any) -> Dict:
        """
        Generate documentation summary for a dependency graph.

        Args:
            graph: DependencyGraph or dict representation

        Returns:
            Dictionary with summary stats and descriptions
        """
        if isinstance(graph, dict):
            # Convert dict back to DependencyGraph if needed
            try:
                from dataclasses import asdict
                # For now, work with dict directly
                modules = len(graph.get("module_hierarchy", {}).get("modules", {}))
                instantiations = graph.get("module_hierarchy", {}).get("total_instances", 0)
                includes = graph.get("include_tree", {}).get("total_includes", 0)
                packages = len(graph.get("package_imports", {}).get("package_defs", {}))
                cycles = len(graph.get("module_hierarchy", {}).get("cycles", []))
                unresolved_includes = len(graph.get("include_tree", {}).get("unresolved_includes", []))
                unresolved_packages = len(graph.get("package_imports", {}).get("unresolved_packages", set()))
            except Exception:
                modules = instantiations = includes = packages = 0
                cycles = unresolved_includes = unresolved_packages = 0
        else:
            # DependencyGraph object
            modules = graph.total_modules
            instantiations = graph.module_hierarchy.total_instances
            includes = graph.total_includes
            packages = graph.total_packages
            cycles = len(graph.module_hierarchy.cycles)
            unresolved_includes = len(graph.include_tree.unresolved_includes)
            unresolved_packages = len(graph.package_imports.unresolved_packages)

        issues = []
        if cycles > 0:
            issues.append(f"Found {cycles} module instantiation cycles")
        if unresolved_includes > 0:
            issues.append(f"Found {unresolved_includes} unresolved includes")
        if unresolved_packages > 0:
            issues.append(f"Found {unresolved_packages} unresolved packages")

        hierarchy_desc = "Flat design (no hierarchy)" if modules <= 1 else \
                        f"Hierarchy with {modules} modules and {instantiations} instantiations"

        return {
            "summary": {
                "modules": modules,
                "instantiations": instantiations,
                "includes": includes,
                "packages": packages,
                "hierarchy_depth": graph.module_hierarchy.max_depth if not isinstance(graph, dict) else 0,
            },
            "issues": issues,
            "hierarchy_description": hierarchy_desc,
            "has_cycles": cycles > 0,
            "has_unresolved": unresolved_includes > 0 or unresolved_packages > 0,
        }

    def propose_modularization(self, graph: Any) -> Dict:
        """
        Suggest modularization improvements.

        Analyzes high fan-out modules, deep hierarchy, circular dependencies.

        Args:
            graph: DependencyGraph or dict representation

        Returns:
            Dictionary with recommendations
        """
        recommendations = []

        # Extract data
        if isinstance(graph, dict):
            modules = graph.get("module_hierarchy", {}).get("modules", {})
            max_depth = graph.get("module_hierarchy", {}).get("max_depth", 0)
            cycles = graph.get("module_hierarchy", {}).get("cycles", [])
        else:
            modules = graph.module_hierarchy.modules
            max_depth = graph.module_hierarchy.max_depth
            cycles = graph.module_hierarchy.cycles

        # Check for high fan-out modules
        high_fanout = []
        for name, module in modules.items() if isinstance(modules, dict) else []:
            fan_out = module.get("fan_out", 0) if isinstance(module, dict) else module.fan_out
            if fan_out > 5:
                high_fanout.append((name, fan_out))

        if high_fanout:
            high_fanout.sort(key=lambda x: x[1], reverse=True)
            top_modules = [f"{m[0]} ({m[1]} children)" for m in high_fanout[:3]]
            recommendations.append({
                "category": "high_fan_out",
                "severity": "medium",
                "description": "Modules with many children may benefit from refactoring",
                "modules": top_modules,
            })

        # Check for deep hierarchy
        if max_depth > 5:
            recommendations.append({
                "category": "deep_hierarchy",
                "severity": "medium",
                "description": f"Module hierarchy is {max_depth} levels deep; consider flattening",
                "max_depth": max_depth,
            })

        # Check for cycles
        if cycles:
            recommendations.append({
                "category": "circular_dependencies",
                "severity": "high",
                "description": f"Found {len(cycles)} circular dependencies; refactor to break cycles",
                "cycle_count": len(cycles),
            })

        return {
            "recommendations": recommendations,
            "priority_actions": [r["category"] for r in recommendations if r.get("severity") == "high"],
        }

    def validate_modularization(self, graph: Any, plan: Optional[Dict] = None) -> Dict:
        """
        Validate modularization quality.

        Checks for cycles, excessive depth, orphan modules.

        Args:
            graph: DependencyGraph or dict representation
            plan: Optional modularization plan (unused for now)

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if isinstance(graph, dict):
            modules = graph.get("module_hierarchy", {}).get("modules", {})
            instantiations = graph.get("module_hierarchy", {}).get("total_instances", 0)
            cycles = graph.get("module_hierarchy", {}).get("cycles", [])
            max_depth = graph.get("module_hierarchy", {}).get("max_depth", 0)
        else:
            modules = graph.module_hierarchy.modules
            instantiations = graph.module_hierarchy.total_instances
            cycles = graph.module_hierarchy.cycles
            max_depth = graph.module_hierarchy.max_depth

        # Cycles are errors
        if cycles:
            results["valid"] = False
            results["errors"].append(f"Found {len(cycles)} circular dependencies")

        # Deep hierarchy is a warning
        if max_depth > 10:
            results["warnings"].append(f"Hierarchy depth {max_depth} exceeds recommended limit of 10")

        # Check for orphans (modules with no parents and no children)
        orphans = []
        if isinstance(modules, dict):
            for name, module in modules.items():
                fan_in = module.get("fan_in", 0) if isinstance(module, dict) else module.fan_in
                fan_out = module.get("fan_out", 0) if isinstance(module, dict) else module.fan_out
                if fan_in == 0 and fan_out == 0:
                    orphans.append(name)
        if orphans:
            results["warnings"].append(f"Found {len(orphans)} orphan modules with no hierarchy connection")

        return results

    def get_module_dependencies(self, file_path: str) -> Dict:
        """
        Get dependencies for a single file/module.

        Returns modules instantiated, files included, packages imported.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with dependencies
        """
        return {
            "file_path": file_path,
            "instantiations": [],
            "includes": [],
            "imports": [],
            "note": "Single-file dependency query requires full graph context; run analyze() first",
        }


# Backward compatibility alias
DependencyAnalyzer = HDLDependencyAnalyzer
