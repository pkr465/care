"""
Optional Verible integration for enhanced HDL parsing accuracy.

Wraps `verible-verilog-syntax --printtree` for AST-level module and
instantiation extraction. Falls back gracefully when Verible is unavailable.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VeribleParserWrapper:
    """Wrapper around verible-verilog-syntax for structured HDL parsing."""

    def __init__(self, executable: str = "verible-verilog-syntax", timeout: int = 30):
        self.executable = executable
        self.timeout = timeout
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.executable) is not None
            if self._available:
                logger.info("Verible parser available: %s", self.executable)
            else:
                logger.info("Verible parser not found; regex fallback will be used")
        return self._available

    def parse_file(self, file_path: str) -> Optional[str]:
        """Run verible-verilog-syntax --printtree and return raw output.
        Returns None if Verible unavailable or parse fails."""
        if not self.available:
            return None
        try:
            result = subprocess.run(
                [self.executable, "--printtree", file_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode == 0:
                return result.stdout
            logger.debug("Verible parse returned %d for %s: %s",
                        result.returncode, file_path, result.stderr[:200])
            return result.stdout  # partial parse may still be useful
        except subprocess.TimeoutExpired:
            logger.warning("Verible timed out on %s", file_path)
            return None
        except Exception as e:
            logger.warning("Verible error on %s: %s", file_path, e)
            return None

    def extract_modules(self, tree_output: str) -> List[Dict[str, Any]]:
        """Extract module names and line numbers from printtree output.
        Returns list of {name, line} dicts."""
        modules = []
        # Verible printtree format: look for kModuleDeclaration nodes
        for match in re.finditer(
            r'kModuleDeclaration.*?kModuleHeader.*?'
            r'SymbolIdentifier\s*@(\d+):\d+-\d+:\d+\s*"(\w+)"',
            tree_output, re.DOTALL
        ):
            line = int(match.group(1))
            name = match.group(2)
            modules.append({"name": name, "line": line})
        return modules

    def extract_instantiations(self, tree_output: str) -> List[Tuple[str, str, int]]:
        """Extract module instantiations from printtree output.
        Returns list of (module_type, instance_name, line) tuples."""
        instantiations = []
        # Look for kDataDeclaration or kGateInstance patterns
        for match in re.finditer(
            r'kInstantiationType.*?SymbolIdentifier\s*@(\d+):\d+-\d+:\d+\s*"(\w+)"'
            r'.*?kGateInstance.*?SymbolIdentifier\s*@\d+:\d+-\d+:\d+\s*"(\w+)"',
            tree_output, re.DOTALL
        ):
            line = int(match.group(1))
            module_type = match.group(2)
            instance_name = match.group(3)
            instantiations.append((module_type, instance_name, line))
        return instantiations
