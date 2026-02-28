"""
Interface Binding Analyzer — SystemVerilog Interface & Modport Analysis.

Analyzes SystemVerilog interface definitions, modport declarations, and interface
instantiations within modules. Builds a complete map of interface bindings and
detects unconnected interfaces.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from . import InterfaceBindingMap, InterfaceDefinition, InterfaceBinding, ModportDef, PortDef

logger = logging.getLogger(__name__)


class InterfaceBindingAnalyzer:
    """
    Analyzes HDL files for interface definitions, modport declarations, and bindings.

    Detects:
    - Interface definitions with their parameters, modports, and signals
    - Modport declarations within interfaces
    - Interface instantiations in modules
    - Modport-qualified interface bindings (interface.modport)
    - Unconnected interface definitions
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the analyzer.

        Args:
            config: Configuration dict with keys:
                - project_root: Root directory of the HDL project
                - debug: Whether to log verbose debug messages
        """
        self.project_root = config.get("project_root", "")
        self.debug = config.get("debug", False)
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

    def _strip_comments(self, source: str) -> str:
        """
        Remove SystemVerilog comments from source code.

        Strips single-line (//) and multi-line (/* */) comments.

        Args:
            source: HDL source code

        Returns:
            Source code with comments removed
        """
        # Remove multi-line comments
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        # Remove single-line comments
        source = re.sub(r"//.*?$", "", source, flags=re.MULTILINE)
        return source

    def _extract_interface_definitions(
        self, source: str, file_path: str
    ) -> Dict[str, InterfaceDefinition]:
        """
        Extract interface definitions from source code.

        Pattern: `interface name ... endinterface`

        Within each interface, extracts:
        - Modport declarations
        - Parameters
        - Signals (logic, wire, reg, bit)

        Args:
            source: Comment-stripped HDL source code
            file_path: Path to the source file

        Returns:
            Dict mapping interface name -> InterfaceDefinition
        """
        interfaces: Dict[str, InterfaceDefinition] = {}

        # Find all interface definitions
        interface_pattern = r"^\s*interface\s+(\w+)\s*(?:#\s*\((.*?)\))?\s*;\s*\n(.*?)^\s*endinterface"
        matches = re.finditer(interface_pattern, source, re.MULTILINE | re.DOTALL)

        for match in matches:
            interface_name = match.group(1)
            params_str = match.group(2) or ""
            interface_body = match.group(3)

            line_no = source[: match.start()].count("\n") + 1

            # Extract parameters
            parameters = self._extract_parameters(params_str, file_path, line_no)

            # Extract modports
            modports = self._extract_modports(interface_body, interface_name, file_path)

            # Extract signals
            signals = self._extract_signals(interface_body)

            interfaces[interface_name] = InterfaceDefinition(
                name=interface_name,
                file_path=file_path,
                line=line_no,
                modports=modports,
                parameters=parameters,
                signals=signals,
            )

            if self.debug:
                self.logger.debug(
                    f"Found interface '{interface_name}' in {file_path}:{line_no} "
                    f"({len(modports)} modports, {len(signals)} signals)"
                )

        return interfaces

    def _extract_modports(
        self, interface_body: str, interface_name: str, file_path: str
    ) -> List[ModportDef]:
        """
        Extract modport declarations from interface body.

        Pattern: `modport name (...ports...)`

        Args:
            interface_body: Source code of interface body
            interface_name: Name of parent interface
            file_path: Path to the file

        Returns:
            List of ModportDef objects
        """
        modports: List[ModportDef] = []

        # Match modport declarations
        modport_pattern = r"modport\s+(\w+)\s*\((.*?)\)"
        matches = re.finditer(modport_pattern, interface_body, re.DOTALL)

        for match in matches:
            modport_name = match.group(1)
            ports_str = match.group(2)

            line_no = interface_body[: match.start()].count("\n") + 1

            # Extract ports within modport
            ports = self._extract_modport_ports(ports_str, file_path, line_no)

            modports.append(
                ModportDef(
                    name=modport_name,
                    interface_name=interface_name,
                    ports=ports,
                    file_path=file_path,
                    line=line_no,
                )
            )

            if self.debug:
                self.logger.debug(
                    f"  Found modport '{modport_name}' in {interface_name} "
                    f"with {len(ports)} ports"
                )

        return modports

    def _extract_modport_ports(self, ports_str: str, file_path: str, line_no: int) -> List[PortDef]:
        """
        Extract port definitions from within a modport declaration.

        Pattern: `(input|output|inout) [type] name`

        Args:
            ports_str: Comma-separated ports string
            file_path: Path to the file
            line_no: Line number

        Returns:
            List of PortDef objects
        """
        ports: List[PortDef] = []

        # Split by commas, but be careful of array dimensions
        port_entries = re.split(r",(?![^\[]*\])", ports_str)

        for entry in port_entries:
            entry = entry.strip()
            if not entry:
                continue

            # Match direction and optional data type
            port_match = re.match(
                r"(input|output|inout|ref)\s+(?:(\w+)\s+)?(?:(\[.*?\]\s*))?(\w+)", entry
            )
            if port_match:
                direction = port_match.group(1)
                data_type = port_match.group(2) or "logic"
                width = port_match.group(3) or ""
                name = port_match.group(4)

                ports.append(
                    PortDef(
                        name=name,
                        direction=direction,
                        data_type=data_type,
                        width=width.strip(),
                        file_path=file_path,
                        line=line_no,
                    )
                )

        return ports

    def _extract_parameters(
        self, params_str: str, file_path: str, line_no: int
    ) -> List:
        """
        Extract parameter declarations.

        Pattern: `parameter type name = value`

        Args:
            params_str: Parameter declaration string
            file_path: Path to the file
            line_no: Line number

        Returns:
            List of parameter definitions
        """
        parameters = []

        if not params_str:
            return parameters

        # Match parameter declarations
        param_pattern = r"(\w+)\s*(?:=\s*([^,\)]+))?"
        matches = re.finditer(param_pattern, params_str)

        for match in matches:
            param_name = match.group(1)
            default_value = match.group(2) or ""

            parameters.append(
                {
                    "name": param_name,
                    "default_value": default_value.strip(),
                    "file_path": file_path,
                    "line": line_no,
                }
            )

        return parameters

    def _extract_signals(self, interface_body: str) -> List[str]:
        """
        Extract signal declarations from interface body.

        Pattern: `(logic|wire|reg|bit) [dimensions] name`

        Args:
            interface_body: Source code of interface body

        Returns:
            List of signal names
        """
        signals: List[str] = []

        # Match signal declarations
        signal_pattern = r"(logic|wire|reg|bit)\s+(?:\[.*?\]\s*)?(\w+)"
        matches = re.finditer(signal_pattern, interface_body)

        for match in matches:
            signal_name = match.group(2)
            signals.append(signal_name)

        return signals

    def _extract_instantiations(
        self, source: str, file_path: str, known_interfaces: Set[str]
    ) -> List[InterfaceBinding]:
        """
        Extract interface instantiations from module source.

        Detects:
        1. Direct instantiation: `interface_name instance_name`
        2. Modport-qualified: `interface_name.modport_name instance_name`

        Args:
            source: Comment-stripped HDL source code
            file_path: Path to the source file
            known_interfaces: Set of known interface names

        Returns:
            List of InterfaceBinding objects
        """
        bindings: List[InterfaceBinding] = []

        # Find module declarations to establish parent scope
        module_pattern = r"^\s*module\s+(\w+)"
        modules = list(re.finditer(module_pattern, source, re.MULTILINE))

        if not modules:
            return bindings

        for i, module_match in enumerate(modules):
            parent_module = module_match.group(1)
            module_start = module_match.start()
            module_end = modules[i + 1].start() if i + 1 < len(modules) else len(source)
            module_body = source[module_start:module_end]

            # Look for interface instantiations (direct and modport-qualified)
            # Direct: interface_name instance_name (
            direct_pattern = r"(\w+)\s+(\w+)\s*\("
            matches = re.finditer(direct_pattern, module_body)

            for match in matches:
                potential_interface = match.group(1)
                instance_name = match.group(2)

                # Check if this is a known interface name
                if potential_interface in known_interfaces:
                    line_no = module_body[: match.start()].count("\n") + 1

                    bindings.append(
                        InterfaceBinding(
                            interface_name=potential_interface,
                            instance_name=instance_name,
                            modport_name="",
                            parent_module=parent_module,
                            file_path=file_path,
                            line=line_no,
                        )
                    )

                    if self.debug:
                        self.logger.debug(
                            f"Found interface binding: {parent_module}.{instance_name} "
                            f"-> {potential_interface}"
                        )

            # Modport-qualified: interface_name.modport_name instance_name
            modport_pattern = r"(\w+)\.(\w+)\s+(\w+)"
            matches = re.finditer(modport_pattern, module_body)

            for match in matches:
                potential_interface = match.group(1)
                modport_name = match.group(2)
                instance_name = match.group(3)

                if potential_interface in known_interfaces:
                    line_no = module_body[: match.start()].count("\n") + 1

                    bindings.append(
                        InterfaceBinding(
                            interface_name=potential_interface,
                            instance_name=instance_name,
                            modport_name=modport_name,
                            parent_module=parent_module,
                            file_path=file_path,
                            line=line_no,
                        )
                    )

                    if self.debug:
                        self.logger.debug(
                            f"Found modport binding: {parent_module}.{instance_name} "
                            f"-> {potential_interface}.{modport_name}"
                        )

        return bindings

    def build(self, file_cache: List[Dict]) -> InterfaceBindingMap:
        """
        Build complete interface binding map from file cache.

        Args:
            file_cache: List of file dicts with keys:
                - file_relative_path or file_path: file path
                - source: HDL source code (may include comments)

        Returns:
            InterfaceBindingMap with all interface definitions, bindings, and analysis
        """
        result = InterfaceBindingMap()

        # First pass: collect all interface definitions
        all_interfaces: Dict[str, InterfaceDefinition] = {}
        known_interface_names: Set[str] = set()

        for file_entry in file_cache:
            # Normalize file_cache keys
            file_path = file_entry.get("file_relative_path") or file_entry.get("file_path") or file_entry.get("path", "")
            source = file_entry.get("source", "")

            if not source:
                continue

            # Strip comments before analysis
            clean_source = self._strip_comments(source)

            # Extract interface definitions
            interfaces = self._extract_interface_definitions(clean_source, file_path)
            all_interfaces.update(interfaces)
            known_interface_names.update(interfaces.keys())

        # Store interface definitions
        result.interface_defs = all_interfaces
        result.total_interfaces = len(all_interfaces)

        # Second pass: find instantiations and bindings
        all_bindings: List[InterfaceBinding] = []
        instantiated_interfaces: Set[str] = set()

        for file_entry in file_cache:
            # Normalize file_cache keys
            file_path = file_entry.get("file_relative_path") or file_entry.get("file_path") or file_entry.get("path", "")
            source = file_entry.get("source", "")

            if not source:
                continue

            clean_source = self._strip_comments(source)

            # Extract bindings
            bindings = self._extract_instantiations(clean_source, file_path, known_interface_names)
            all_bindings.extend(bindings)

            # Track which interfaces are instantiated
            for binding in bindings:
                instantiated_interfaces.add(binding.interface_name)

        result.bindings = all_bindings
        result.total_bindings = len(all_bindings)

        # Track modport usage
        modport_usage: Dict[str, List[str]] = {}
        for binding in all_bindings:
            if binding.modport_name:
                key = f"{binding.interface_name}.{binding.modport_name}"
                if key not in modport_usage:
                    modport_usage[key] = []
                modport_usage[key].append(binding.parent_module)

        result.modport_usage = modport_usage

        # Detect unconnected interfaces
        unconnected = [
            name
            for name in all_interfaces.keys()
            if name not in instantiated_interfaces
        ]
        result.unconnected_interfaces = unconnected

        if self.debug:
            self.logger.debug(
                f"Interface binding analysis complete: "
                f"{result.total_interfaces} interfaces, "
                f"{result.total_bindings} bindings, "
                f"{len(unconnected)} unconnected"
            )

        return result
