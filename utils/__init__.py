"""
utils - Shared utilities for the Codebase Analysis & Refactor Engine.

Subpackages:
    utils.common   - Standalone tool classes (LLM, Email, Excel, Mermaid)
    utils.parsers  - Configuration and data parsers (EnvConfig)

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    utils                             │
    │  ┌──────────────────┐  ┌──────────────────────────┐ │
    │  │   utils.parsers   │  │     utils.common         │ │
    │  │  ┌──────────────┐ │  │  ┌──────────────────────┐│ │
    │  │  │ EnvConfig    │ │  │  │ LLMTools (Claude)    ││ │
    │  │  └──────────────┘ │  │  │ EmailReporter (SMTP) ││ │
    │  └──────────────────┘  │  │ ExcelWriter (openpyxl)││ │
    │                         │  │ MermaidConverter      ││ │
    │                         │  └──────────────────────┘│ │
    │                         └──────────────────────────┘ │
    └─────────────────────────────────────────────────────┘
"""

__version__ = "2.0.0"
