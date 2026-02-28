"""
Adapter layer for HDL static analysis.

Adapters bridge Verible (SystemVerilog parser/linter), Verilator, and Yosys
into the health report pipeline. MetricsCalculator orchestrates them.
"""

from agents.adapters.base_adapter import BaseStaticAdapter

# --- Dead Code / Unused Module Adapter ---
try:
    from agents.adapters.dead_code_adapter import UnusedModuleAdapter
    UnusedSignalAdapter = UnusedModuleAdapter  # backward-compat alias
except ImportError:
    UnusedModuleAdapter = None
    UnusedSignalAdapter = None

# --- AST Complexity / HDL Complexity Adapter ---
try:
    from agents.adapters.ast_complexity_adapter import HDLComplexityAdapter
    HierarchyComplexityAdapter = HDLComplexityAdapter  # backward-compat alias
except ImportError:
    HDLComplexityAdapter = None
    HierarchyComplexityAdapter = None

# --- Security / Lint Adapter ---
try:
    from agents.adapters.security_adapter import LintAdapter
    DesignRuleAdapter = LintAdapter  # backward-compat alias
except ImportError:
    LintAdapter = None
    DesignRuleAdapter = None

# --- Call Graph / Hierarchy Analyzer Adapter ---
try:
    from agents.adapters.call_graph_adapter import HierarchyAnalyzerAdapter
    ModuleGraphAdapter = HierarchyAnalyzerAdapter  # backward-compat alias
except ImportError:
    HierarchyAnalyzerAdapter = None
    ModuleGraphAdapter = None

# --- Function Metrics / Module Metrics Adapter ---
try:
    from agents.adapters.function_metrics_adapter import ModuleMetricsAdapter
    PortMetricsAdapter = ModuleMetricsAdapter  # backward-compat alias
except ImportError:
    ModuleMetricsAdapter = None
    PortMetricsAdapter = None

# --- Excel Report Adapter ---
try:
    from agents.adapters.excel_report_adapter import ExcelReportAdapter
except ImportError:
    ExcelReportAdapter = None

# --- Dependency Graph Adapter (NEW) ---
try:
    from agents.adapters.dependency_graph_adapter import DependencyGraphAdapter
except ImportError:
    DependencyGraphAdapter = None

__all__ = [
    "BaseStaticAdapter",
    # Current names
    "UnusedModuleAdapter",
    "HDLComplexityAdapter",
    "LintAdapter",
    "HierarchyAnalyzerAdapter",
    "ModuleMetricsAdapter",
    "ExcelReportAdapter",
    "DependencyGraphAdapter",
    # Backward-compat aliases
    "UnusedSignalAdapter",
    "HierarchyComplexityAdapter",
    "DesignRuleAdapter",
    "ModuleGraphAdapter",
    "PortMetricsAdapter",
]
