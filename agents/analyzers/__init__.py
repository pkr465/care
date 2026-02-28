"""
Specialized analyzers for HDL (Verilog/SystemVerilog) design metrics.
"""

from .quality_analyzer import QualityAnalyzer
from .complexity_analyzer import ComplexityAnalyzer
from .documentation_analyzer import DocumentationAnalyzer
from .maintainability_analyzer import MaintainabilityAnalyzer

# Optional analyzers — graceful degradation if not present
try:
    from .dependency_analyzer import HDLDependencyAnalyzer, DependencyAnalyzer, AnalyzerConfig
except ImportError:
    HDLDependencyAnalyzer = None
    DependencyAnalyzer = None
    AnalyzerConfig = None

__all__ = [
    'QualityAnalyzer',
    'ComplexityAnalyzer',
    'DocumentationAnalyzer',
    'MaintainabilityAnalyzer',
    'HDLDependencyAnalyzer',
    'DependencyAnalyzer',
    'AnalyzerConfig',
]
