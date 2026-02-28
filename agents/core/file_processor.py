"""
Verilog/SystemVerilog file discovery, processing, and language detection (enhanced)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import Counter
import fnmatch


class FileProcessor:
    """
    Handles Verilog/SystemVerilog file discovery, processing, and basic analysis.

    Supports:
    - Verilog source files (.v)
    - SystemVerilog source files (.sv, .svh)
    - Verilog headers (.vh)
    - VHDL files (.vhd, .vhdl) - optional support
    """

    # Verilog/SystemVerilog file extensions
    VERILOG_SOURCE_EXTS = {".v"}
    SYSTEMVERILOG_SOURCE_EXTS = {".sv", ".svh"}
    VERILOG_HEADER_EXTS = {".vh"}
    VHDL_EXTS = {".vhd", ".vhdl"}

    ALL_HDL_EXTS = (
        VERILOG_SOURCE_EXTS
        | SYSTEMVERILOG_SOURCE_EXTS
        | VERILOG_HEADER_EXTS
        | VHDL_EXTS
    )

    # Default exclusions for HDL projects
    DEFAULT_EXCLUDE_DIRS = {
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        "build",
        "sim_results",
        "synthesis",
        "implementation",
        ".Xil",
        "work",
        "xsim.dir",
        "vivado",
        "eda",
        "third_party",
        "external",
        "vendor",
        ".idea",
        ".vscode",
        ".vs",
        "__pycache__",
        ".pytest_cache",
        "constraints",
        ".ipynb_checkpoints",
    }

    DEFAULT_EXCLUDE_GLOBS = [
        "*.vcd",
        "*.wlf",
        "*.fsdb",
        "*.vpd",
        "*.saif",
        "*.log",
        "*.tmp",
        "*.temp",
        "*.zip",
        "*.tar",
        "*.gz",
        "*.bz2",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.svg",
        "*.pdf",
        "*.doc",
        "*.docx",
        "*.xdc",  # Xilinx constraint files
        "*.sdc",  # Synopsys constraint files
        "*.pdc",  # Lattice constraint files
        "*_tb.sv",  # Testbench exclusion (optional)
        "*_tb.v",
    ]

    def __init__(
        self,
        codebase_path: str,
        max_files: int = 10000,
        exclude_dirs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
    ):
        """
        Initialize Verilog/SystemVerilog file processor.

        Args:
            codebase_path: Path to HDL codebase root
            max_files: Maximum files to process
            exclude_dirs: Additional directories to exclude (names, not paths)
            exclude_globs: Additional glob patterns to exclude (relative paths)
        """
        self.codebase_path = Path(codebase_path).resolve()
        self.max_files = max_files

        # Exclusions: use sets/lists for efficient checks
        self.exclude_dirs = self.DEFAULT_EXCLUDE_DIRS | set(exclude_dirs or [])
        self.exclude_globs = self.DEFAULT_EXCLUDE_GLOBS + (exclude_globs or [])

        # Cache
        self._file_cache: List[Dict[str, Any]] = []
        self._language_stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be excluded based on directory names and glob patterns.

        Only inspects path components *relative* to the codebase root so that
        the location of the codebase itself (e.g. inside an ``out/`` directory)
        does not accidentally trigger an exclusion.
        """
        # Relative path for glob matching; guard against paths outside codebase
        try:
            rel_path = path.relative_to(self.codebase_path)
        except ValueError:
            # Path not under codebase root (e.g., symlink); treat as excluded
            return True

        # Directory exclusions — only check parts *below* codebase root
        for part in rel_path.parts:
            if part in self.exclude_dirs:
                return True

        rel_path_str = rel_path.as_posix().lower()
        for pattern in self.exclude_globs:
            if fnmatch.fnmatch(rel_path_str, pattern.lower()):
                return True

        return False

    def _detect_language(self, file_path: Path, content: str = "") -> str:
        """
        Detect HDL language variant from file extension and (optionally) content.

        Returns: "verilog", "systemverilog", "vhdl", or "unknown"
        """
        suffix = file_path.suffix.lower()

        if suffix in self.VHDL_EXTS:
            return "vhdl"

        if suffix in self.VERILOG_HEADER_EXTS:
            return "verilog_header"

        if suffix in self.VERILOG_SOURCE_EXTS:
            # Use content to distinguish Verilog vs SystemVerilog
            if content and self._is_systemverilog_content(content):
                return "systemverilog"
            return "verilog"

        if suffix in self.SYSTEMVERILOG_SOURCE_EXTS:
            return "systemverilog"

        return "unknown"

    def _is_systemverilog_content(self, content: str) -> bool:
        """Analyze content to determine if it's SystemVerilog rather than Verilog."""
        systemverilog_indicators = [
            r"\bclass\s+\w+",
            r"\binterface\s+\w+",
            r"\bpackage\s+\w+",
            r"\bimport\s+\w+\s*::",
            r"\btypedef\s+",
            r"\bstruct\s+\{",
            r"\benum\s+\{",
            r"\bbit\s+",
            r"\blogic\s+",
            r"\balways_ff\s+@",
            r"\balways_comb\s+",
            r"\balways_latch\s+",
            r"\bunique\s+case",
            r"\bpriority\s+case",
            r"\bassert\s+property",
            r"\bmodport\s+",
            r"\bclocking\s+",
            r"\bprogram\s+\w+",
            r"\bproperty\s+\w+",
            r"\bsequence\s+\w+",
            r"\bcoverage\s+\{",
            r"\brandomize\s*\(",
            r"\bwith\s*\{",
        ]

        for pattern in systemverilog_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _extract_includes(self, content: str) -> List[Dict[str, Any]]:
        """Extract include statements from Verilog/SystemVerilog code.

        Supports:
        - `include "filename"
        - import package_name::*;
        """
        includes: List[Dict[str, Any]] = []

        for line_num, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()

            # Verilog/SystemVerilog includes: `include "filename"
            include_match = re.match(r"^\s*`\s*include\s+['\"]([^'\"]+)['\"]", stripped)
            if include_match:
                includes.append(
                    {
                        "file": include_match.group(1),
                        "line": line_num,
                        "type": "include",
                        "raw_line": line.rstrip("\n"),
                    }
                )
                continue

            # SystemVerilog import: import package_name::*;
            import_match = re.match(r"^\s*import\s+(\w+)\s*::\s*\*\s*;", stripped)
            if import_match:
                includes.append(
                    {
                        "file": import_match.group(1),
                        "line": line_num,
                        "type": "package_import",
                        "raw_line": line.rstrip("\n"),
                    }
                )

        return includes

    def _calculate_basic_metrics(self, content: str, language: str) -> Dict[str, Any]:
        """Calculate basic Verilog/SystemVerilog code metrics."""
        lines = content.splitlines()

        metrics: Dict[str, Any] = {
            "total_lines": len(lines),
            "non_empty_lines": 0,
            "comment_lines": 0,
            "code_lines": 0,
            "compiler_directives": 0,  # `define, `ifdef, etc.
        }

        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Blank line: counted only in total_lines
                continue

            metrics["non_empty_lines"] += 1

            # Multi-line comment continuation
            if in_multiline_comment:
                metrics["comment_lines"] += 1
                if "*/" in stripped:
                    in_multiline_comment = False
                continue

            # Multi-line comment start
            if stripped.startswith("/*"):
                metrics["comment_lines"] += 1
                if "*/" not in stripped:
                    in_multiline_comment = True
                continue

            # Single line comments
            if stripped.startswith("//"):
                metrics["comment_lines"] += 1
                continue

            # Compiler directives (`define, `ifdef, `include, etc.)
            if stripped.startswith("`"):
                metrics["compiler_directives"] += 1
                continue

            # Everything else is code
            metrics["code_lines"] += 1

        total_non_empty = max(1, metrics["non_empty_lines"])
        metrics["comment_ratio"] = metrics["comment_lines"] / total_non_empty
        metrics["code_ratio"] = metrics["code_lines"] / total_non_empty
        metrics["directive_ratio"] = metrics["compiler_directives"] / total_non_empty

        return metrics

    def _extract_modules(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract module, interface, and package definitions from Verilog/SystemVerilog.

        This is a simplified heuristic regex-based extractor, not a full parser.
        """
        modules: List[Dict[str, Any]] = []

        # Module definitions
        module_pattern = r"""
            (?:^|\n)                          # Start of line
            \s*                               # Optional whitespace
            module\s+                         # 'module' keyword
            (\w+)                             # Module name (capture group 1)
            (?:\s*#\s*\(([^)]*)\))?          # Optional parameters (capture group 2)
            \s*\(([^)]*)\)                    # Port list (capture group 3)
            \s*;                              # Semicolon
        """

        for match in re.finditer(module_pattern, content, re.VERBOSE | re.MULTILINE):
            module_name = match.group(1)
            parameters = match.group(2) or ""
            ports = match.group(3) or ""

            line_num = content[: match.start()].count("\n") + 1

            port_count = len([p for p in ports.split(",") if p.strip()])
            param_count = len([p for p in parameters.split(",") if p.strip()])

            modules.append(
                {
                    "name": module_name,
                    "type": "module",
                    "line": line_num,
                    "parameters": parameters,
                    "parameter_count": param_count,
                    "ports": ports,
                    "port_count": port_count,
                }
            )

        # Interface definitions (SystemVerilog)
        interface_pattern = r"""
            (?:^|\n)                          # Start of line
            \s*                               # Optional whitespace
            interface\s+                      # 'interface' keyword
            (\w+)                             # Interface name (capture group 1)
            (?:\s*#\s*\(([^)]*)\))?          # Optional parameters (capture group 2)
            \s*;                              # Semicolon
        """

        for match in re.finditer(interface_pattern, content, re.VERBOSE | re.MULTILINE):
            interface_name = match.group(1)
            parameters = match.group(2) or ""

            line_num = content[: match.start()].count("\n") + 1

            param_count = len([p for p in parameters.split(",") if p.strip()])

            modules.append(
                {
                    "name": interface_name,
                    "type": "interface",
                    "line": line_num,
                    "parameters": parameters,
                    "parameter_count": param_count,
                }
            )

        # Package definitions (SystemVerilog)
        package_pattern = r"""
            (?:^|\n)                          # Start of line
            \s*                               # Optional whitespace
            package\s+                        # 'package' keyword
            (\w+)                             # Package name (capture group 1)
            \s*;                              # Semicolon
        """

        for match in re.finditer(package_pattern, content, re.VERBOSE | re.MULTILINE):
            package_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            modules.append(
                {
                    "name": package_name,
                    "type": "package",
                    "line": line_num,
                }
            )

        # Task and function definitions
        task_func_pattern = r"""
            (?:^|\n)                          # Start of line
            \s*                               # Optional whitespace
            (?:static\s+)?                    # Optional static
            (task|function)\s+                # 'task' or 'function' keyword
            (?:\w+\s+)?                       # Optional return type (for functions)
            (\w+)                             # Task/function name (capture group 2)
            \s*\(([^)]*)\)                    # Port/parameter list (capture group 3)
            \s*;                              # Semicolon
        """

        for match in re.finditer(task_func_pattern, content, re.VERBOSE | re.MULTILINE):
            def_type = match.group(1)
            name = match.group(2)
            params = match.group(3) or ""

            line_num = content[: match.start()].count("\n") + 1
            param_count = len([p for p in params.split(",") if p.strip()])

            modules.append(
                {
                    "name": name,
                    "type": def_type,
                    "line": line_num,
                    "parameters": params,
                    "parameter_count": param_count,
                }
            )

        return modules

    # ------------------------------------------------------------------ #
    # Main processing
    # ------------------------------------------------------------------ #

    def process_files(self) -> List[Dict[str, Any]]:
        """Process all Verilog/SystemVerilog/VHDL files in the codebase and populate file cache."""
        if not self.codebase_path.exists():
            raise FileNotFoundError(
                f"Codebase path does not exist: {self.codebase_path}"
            )

        # Reset caches for repeatability
        self._file_cache = []
        self._language_stats = {}

        files: List[Dict[str, Any]] = []
        processed_count = 0

        print(f"Processing HDL files in: {self.codebase_path}")
        print(f"Supported extensions: {sorted(self.ALL_HDL_EXTS)}")

        for root, dirs, filenames in os.walk(self.codebase_path):
            # Filter out excluded directories (by name)
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for filename in filenames:
                if processed_count >= self.max_files:
                    break

                file_path = Path(root) / filename

                # Check exclusions (dirs, globs, and out-of-tree paths)
                if self._is_excluded(file_path):
                    continue

                suffix = file_path.suffix.lower()
                if suffix not in self.ALL_HDL_EXTS:
                    continue

                # Read file content
                try:
                    with open(
                        file_path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        content = f.read()
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue

                # Detect language
                language = self._detect_language(file_path, content)
                if language == "unknown":
                    continue

                # Relative path from codebase root
                try:
                    rel_path = file_path.relative_to(self.codebase_path)
                except ValueError:
                    # Should not normally happen, but skip if outside root
                    continue

                # Extract includes
                includes = self._extract_includes(content)

                # Calculate basic metrics
                metrics = self._calculate_basic_metrics(content, language)

                # Extract modules/interfaces/packages
                modules: List[Dict[str, Any]] = []
                if language in ["verilog", "systemverilog"]:
                    modules = self._extract_modules(content, language)

                # Build file entry
                file_entry: Dict[str, Any] = {
                    "path_obj": file_path,
                    "file_name": file_path.name,
                    "file_relative_path": rel_path.as_posix(),
                    "suffix": suffix,
                    "language": language,
                    "source": content,
                    "size_bytes": len(content.encode("utf-8")),
                    "includes": includes,
                    "modules": modules,
                    "metrics": metrics,
                    "module_key": self._generate_module_key(rel_path),
                }

                files.append(file_entry)
                processed_count += 1

                # Update language stats
                self._language_stats[language] = (
                    self._language_stats.get(language, 0) + 1
                )

            if processed_count >= self.max_files:
                break

        self._file_cache = files

        print(f"Processed {len(files)} HDL files")
        print(f"Language distribution: {dict(self._language_stats)}")

        return files

    def _generate_module_key(self, rel_path: Path) -> str:
        """Generate a module key for dependency tracking."""
        path_str = rel_path.as_posix()
        if "." in path_str:
            path_str = path_str.rsplit(".", 1)[0]
        module_key = path_str.replace("/", ".")
        return module_key

    # ------------------------------------------------------------------ #
    # Structure analysis
    # ------------------------------------------------------------------ #

    def analyze_structure(self, file_cache: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze Verilog/SystemVerilog/VHDL codebase structure and provide summary."""
        cache = file_cache or self._file_cache

        if not cache:
            return {"error": "No HDL files processed"}

        # If a custom cache is provided, recompute language stats from it
        lang_counts = Counter(f.get("language", "unknown") for f in cache)
        file_stats = {
            "total_files": len(cache),
            "languages": dict(lang_counts),
            "file_types": Counter(f.get("suffix", "") for f in cache),
            "total_size_bytes": sum(f.get("size_bytes", 0) for f in cache),
        }
        file_stats["avg_file_size"] = (
            file_stats["total_size_bytes"] / file_stats["total_files"]
            if file_stats["total_files"] > 0
            else 0
        )

        # Source vs headers
        source_files = [
            f
            for f in cache
            if f.get("language") in ["verilog", "systemverilog"]
        ]
        header_files = [
            f for f in cache if f.get("language") in ["verilog_header"]
        ]
        vhdl_files = [f for f in cache if f.get("language") == "vhdl"]

        file_stats.update(
            {
                "verilog_systemverilog_files": len(source_files),
                "verilog_header_files": len(header_files),
                "vhdl_files": len(vhdl_files),
                "source_to_header_ratio": len(source_files)
                / max(1, len(header_files)),
            }
        )

        # Code metrics aggregation
        total_lines = sum(f["metrics"]["total_lines"] for f in cache)
        total_code_lines = sum(f["metrics"]["code_lines"] for f in cache)
        total_comment_lines = sum(f["metrics"]["comment_lines"] for f in cache)
        total_preprocessor_lines = sum(
            f["metrics"]["preprocessor_lines"] for f in cache
        )

        code_metrics = {
            "total_lines": total_lines,
            "total_code_lines": total_code_lines,
            "total_comment_lines": total_comment_lines,
            "total_preprocessor_lines": total_preprocessor_lines,
            "overall_comment_ratio": total_comment_lines / max(1, total_lines),
            "overall_code_ratio": total_code_lines / max(1, total_lines),
            "overall_preprocessor_ratio": total_preprocessor_lines
            / max(1, total_lines),
        }

        # Include/Import analysis
        all_includes: List[Dict[str, Any]] = []
        for f in cache:
            all_includes.extend(f.get("includes", []))

        verilog_includes = [inc for inc in all_includes if inc.get("type") == "include"]
        package_imports = [inc for inc in all_includes if inc.get("type") == "package_import"]

        include_stats = {
            "total_includes": len(all_includes),
            "verilog_includes": len(verilog_includes),
            "package_imports": len(package_imports),
        }

        verilog_include_names = [inc.get("file", "") for inc in verilog_includes]
        package_import_names = [inc.get("file", "") for inc in package_imports]

        common_includes = {
            "verilog": Counter(verilog_include_names).most_common(10),
            "packages": Counter(package_import_names).most_common(10),
        }

        # Module/interface analysis
        all_modules: List[Dict[str, Any]] = []
        for f in source_files:
            all_modules.extend(f.get("modules", []))

        module_types = Counter(m.get("type", "unknown") for m in all_modules)

        module_stats: Dict[str, Any] = {
            "total_module_definitions": len(all_modules),
            "avg_modules_per_file": len(all_modules) / max(1, len(source_files)),
            "module_types": dict(module_types),
        }

        if all_modules:
            port_counts = [m.get("port_count", 0) for m in all_modules if m.get("type") == "module"]
            if port_counts:
                module_stats.update(
                    {
                        "avg_ports_per_module": sum(port_counts)
                        / max(1, len(port_counts)),
                        "max_ports": max(port_counts),
                        "min_ports": min(port_counts),
                    }
                )

            param_counts = [m.get("parameter_count", 0) for m in all_modules if "parameter_count" in m]
            if param_counts:
                module_stats.update(
                    {
                        "avg_parameters_per_module": sum(param_counts)
                        / max(1, len(param_counts)),
                        "max_parameters": max(param_counts),
                    }
                )

        # README analysis
        readme_content = self._extract_readme_content()

        # Build system detection
        build_systems = self._detect_build_systems()

        return {
            "file_stats": file_stats,
            "code_metrics": code_metrics,
            "include_stats": include_stats,
            "common_includes": common_includes,
            "module_stats": module_stats,
            "build_systems": build_systems,
            "readme_notes": readme_content,
        }

    def _extract_readme_content(self) -> str:
        """Extract README content if available (first ~1000 characters)."""
        readme_candidates = [
            "README.md",
            "README.rst",
            "README.txt",
            "README",
            "readme.md",
            "readme.rst",
            "readme.txt",
            "readme",
        ]

        for candidate in readme_candidates:
            readme_path = self.codebase_path / candidate
            if readme_path.exists():
                try:
                    content = readme_path.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    return content
                except Exception:
                    continue

        return ""

    def _detect_build_systems(self) -> List[str]:
        """Detect HDL build/synthesis tools used in the codebase (root-level scan)."""
        build_systems: List[str] = []

        build_files = {
            "Makefile": "Make",
            "makefile": "Make",
            "vivado.tcl": "Vivado",
            "build.tcl": "Vivado",
            "quartus_project.qpf": "Quartus",
            "synthesis.tcl": "Synopsys (DC/PT)",
            "synplify.prj": "Synplify",
            "build.xml": "BuildTools",
            "meson.build": "Meson",
            "CMakeLists.txt": "CMake",
        }

        for build_file, system in build_files.items():
            if (self.codebase_path / build_file).exists():
                if system not in build_systems:
                    build_systems.append(system)

        # Vivado projects
        vivado_patterns = ["*.xpr", "*.xprj"]
        for pattern in vivado_patterns:
            if list(self.codebase_path.glob(pattern)):
                if "Vivado" not in build_systems:
                    build_systems.append("Vivado")
                break

        # Quartus projects
        quartus_patterns = ["*.qpf", "*.qsf"]
        for pattern in quartus_patterns:
            if list(self.codebase_path.glob(pattern)):
                if "Quartus" not in build_systems:
                    build_systems.append("Quartus")
                break

        return build_systems

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_file_cache(self) -> List[Dict[str, Any]]:
        """Get processed file cache."""
        return self._file_cache

    def get_language_stats(self) -> Dict[str, int]:
        """Get language distribution statistics from last processing run."""
        return dict(self._language_stats)

    def get_files_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get all files for a specific language."""
        return [f for f in self._file_cache if f.get("language") == language]

    def get_source_files(self) -> List[Dict[str, Any]]:
        """Get all Verilog/SystemVerilog source files."""
        return [f for f in self._file_cache if f.get("language") in ["verilog", "systemverilog"]]

    def get_header_files(self) -> List[Dict[str, Any]]:
        """Get all Verilog header files."""
        return [
            f
            for f in self._file_cache
            if f.get("language") in ["verilog_header"]
        ]

    def get_vhdl_files(self) -> List[Dict[str, Any]]:
        """Get all VHDL files."""
        return [f for f in self._file_cache if f.get("language") == "vhdl"]