"""
Core HDL analysis components.
"""

try:
    from .file_processor import FileProcessor
except ImportError:
    FileProcessor = None

try:
    from .metrics_calculator import MetricsCalculator
except ImportError:
    MetricsCalculator = None

try:
    from .verible_parser_wrapper import VeribleParserWrapper
except ImportError:
    VeribleParserWrapper = None

__all__ = ['FileProcessor', 'MetricsCalculator', 'VeribleParserWrapper']
