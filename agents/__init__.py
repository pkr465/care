"""
CARE — Codebase Analysis & Repair Engine for HDL
Agents Package

Primary agents:
    StaticAnalyzerAgent  — Unified 7-phase pipeline for HDL static analysis and health reporting.
    CodebaseHDLDesignAgent — HDL design analysis agent for Verilog/SystemVerilog.
    CodebaseDesignRepairAgent — Design repair agent (diff original vs patched, report new issues).
"""

from .codebase_static_agent import StaticAnalyzerAgent

# HDL design agents (optional — no hard dependencies)
try:
    from .codebase_hdl_design_agent import CodebaseHDLDesignAgent
    HDL_DESIGN_AGENT_AVAILABLE = True
except ImportError:
    CodebaseHDLDesignAgent = None
    HDL_DESIGN_AGENT_AVAILABLE = False

# Design repair agent (optional — no hard dependencies)
try:
    from .codebase_design_repair_agent import CodebaseDesignRepairAgent
    DESIGN_REPAIR_AGENT_AVAILABLE = True
except ImportError:
    CodebaseDesignRepairAgent = None
    DESIGN_REPAIR_AGENT_AVAILABLE = False

__all__ = [
    'StaticAnalyzerAgent',
    'CodebaseHDLDesignAgent',
    'CodebaseDesignRepairAgent',
    'HDL_DESIGN_AGENT_AVAILABLE',
    'DESIGN_REPAIR_AGENT_AVAILABLE',
]
