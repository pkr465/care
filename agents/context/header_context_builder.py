"""
HDL Context Builder for LLM-Exclusive Verilog/SystemVerilog Analysis.

Resolves `include directives and import packages, parses include files for type definitions
(enums, structs, macros, typedefs, module/interface declarations, parameters, and ports),
and builds chunk-specific context strings to inject into LLM prompts.

Works entirely via regex — no external tooling required.
Caches parsed include files for the lifetime of the builder instance.
"""

import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnumMember:
    name: str
    value: Optional[str] = None          # "0", "0x10", "FOO + 1", or None (auto)
    numeric_value: Optional[int] = None  # Resolved integer if deterministic


@dataclass
class EnumDef:
    name: str                            # enum tag or typedef alias
    members: List[EnumMember] = field(default_factory=list)
    is_typedef: bool = False
    raw: str = ""                        # Compact single-line repr


@dataclass
class StructField:
    type_name: str
    field_name: str
    array_size: Optional[str] = None     # e.g. "64", "MAX_BUF", None


@dataclass
class StructDef:
    name: str
    kind: str = "struct"                 # "struct" or "union"
    fields: List[StructField] = field(default_factory=list)
    is_typedef: bool = False
    raw: str = ""


@dataclass
class MacroDef:
    name: str
    value: str
    is_numeric: bool = False
    numeric_value: Optional[int] = None
    is_function_like: bool = False
    raw: str = ""


@dataclass
class TypedefDef:
    alias: str
    original_type: str
    raw: str = ""


@dataclass
class ModuleDecl:
    name: str
    ports: List[str] = field(default_factory=list)  # List of port declarations
    parameters: List[str] = field(default_factory=list)  # List of parameter names
    raw: str = ""


@dataclass
class InterfaceDecl:
    name: str
    methods: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class TaskProto:
    name: str
    params: str
    raw: str = ""


@dataclass
class FunctionProto:
    name: str
    return_type: str
    params: str
    raw: str = ""


@dataclass
class ParameterDef:
    name: str
    data_type: str
    default_value: Optional[str] = None
    is_localparam: bool = False
    raw: str = ""


@dataclass
class PortDef:
    name: str
    direction: str  # "input", "output", "inout"
    data_type: str
    width: Optional[str] = None
    raw: str = ""


@dataclass
class IncludeDefinitions:
    enums: List[EnumDef] = field(default_factory=list)
    structs: List[StructDef] = field(default_factory=list)
    macros: List[MacroDef] = field(default_factory=list)
    typedefs: List[TypedefDef] = field(default_factory=list)
    parameters: List[ParameterDef] = field(default_factory=list)
    ports: List[PortDef] = field(default_factory=list)
    module_decls: List[ModuleDecl] = field(default_factory=list)
    interface_decls: List[InterfaceDecl] = field(default_factory=list)
    task_protos: List[TaskProto] = field(default_factory=list)
    function_protos: List[FunctionProto] = field(default_factory=list)
    file_path: str = ""


@dataclass
class ResolvedInclude:
    name: str                    # As written in the #include directive
    abs_path: Optional[str]      # Resolved absolute path (None if unresolved)
    include_type: str            # "local" or "system"
    resolved: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Well-known standard SystemVerilog packages (skip for context injection)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PACKAGES: Set[str] = {
    # IEEE SystemVerilog standard packages
    "std", "uvm", "uvm_pkg",
    # Common simulation packages
    "verilog", "verilog_builtins",
    # Xilinx packages
    "xilinx", "xilinx_dma_pkg", "xilinx_axi_pkg",
    # Mentor/Cadence packages
    "mentor", "cadence",
    # Altera/Intel packages
    "altera", "intel", "altera_mf",
    # Common open-source framework packages
    "axi_pkg", "apb_pkg", "ahb_pkg",
}

# Include guard patterns (SystemVerilog: `ifndef, `define, `endif)
_INCLUDE_GUARD_RE = re.compile(
    r"^_+[A-Z][A-Z0-9_]*_SV_*$|^[A-Z][A-Z0-9_]*_SV(?:H)?_*$"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Regex patterns for SystemVerilog/Verilog definition extraction
# ═══════════════════════════════════════════════════════════════════════════════

# Include directive: `include "path/file.svh" or `include "file.vh"
_INCLUDE_RE = re.compile(
    r'^\s*`\s*include\s+["]([^"]+)["]', re.MULTILINE
)

# Import package: import pkg_name::*;
_IMPORT_RE = re.compile(
    r'^\s*import\s+(\w+)(?:::\*)?\s*;', re.MULTILINE
)

# Enum: typedef enum { ... } name or enum type_name { ... };
_ENUM_TYPEDEF_RE = re.compile(
    r'typedef\s+enum\s+(\w+)?\s*\{([^}]*)\}\s*(\w+)\s*;', re.DOTALL
)
_ENUM_PLAIN_RE = re.compile(
    r'(?<!typedef\s)enum\s+(\w+)?\s*\{([^}]*)\}', re.DOTALL
)

# Struct/union: both plain and typedef (SystemVerilog packed structs)
_STRUCT_TYPEDEF_RE = re.compile(
    r'typedef\s+(?:packed\s+)?(struct|union)\s+(\w+)?\s*\{', re.DOTALL
)
_STRUCT_PLAIN_RE = re.compile(
    r'(?<!typedef\s)(?:packed\s+)?(struct|union)\s+(\w+)\s*\{', re.DOTALL
)

# Macro: `define MACRO_NAME value
_MACRO_RE = re.compile(
    r'^\s*`\s*define\s+(\w+)(?:\(([^)]*)\))?\s*(.*?)$', re.MULTILINE
)
_MACRO_CONTINUATION_RE = re.compile(r'\\\s*$')

# Typedef (non-struct/enum)
_TYPEDEF_SIMPLE_RE = re.compile(
    r'typedef\s+(?!enum\b)(?!(?:packed\s+)?struct\b)(?!(?:packed\s+)?union\b)([\w\[\]\s*]+?)\s+(\w+)\s*;'
)

# Parameter: parameter type name = default;  or  localparam type name = default;
_PARAMETER_RE = re.compile(
    r'^\s*(localparam|parameter)\s+([\w\[\]:\s*,]+?)\s+(\w+)(?:\s*=\s*([^;]+))?\s*;',
    re.MULTILINE
)

# Port declarations in module header: input/output/inout type [width] name
_PORT_RE = re.compile(
    r'^\s*(input|output|inout|ref)\s+([\w\s\[\]:*]+?)\s+(\w+)\s*(?:\s*[,;])?$',
    re.MULTILINE
)

# Module declaration: module name(...); or module name #(...) (...);
_MODULE_DECL_RE = re.compile(
    r'^\s*module\s+(\w+)\s*(?:#\s*\(([^)]*)\))?\s*\(([^)]*)\);',
    re.MULTILINE | re.DOTALL
)

# Interface declaration: interface name; ... endinterface
_INTERFACE_DECL_RE = re.compile(
    r'^\s*interface\s+(\w+)\s*(?:#\s*\(([^)]*)\))?.*;',
    re.MULTILINE
)

# Task/function declaration: task/function return_type name(params);
_TASK_PROTO_RE = re.compile(
    r'^\s*task\s+(?:automatic\s+)?(?:([\w\[\]:*]+?)\s+)?(\w+)\s*\(([^)]*)\)\s*;',
    re.MULTILINE
)

_FUNCTION_PROTO_RE = re.compile(
    r'^\s*function\s+(?:automatic\s+)?([\w\[\]:*]+?)\s+(\w+)\s*\(([^)]*)\)\s*;',
    re.MULTILINE
)

# Numeric literal
_NUMERIC_RE = re.compile(
    r'^[+-]?\s*(?:0[xX][0-9a-fA-F]+[uUlL]*|0[bB][01]+[uUlL]*|[0-9]+[uUlL]*)$'
)

# Simple arithmetic expression (for macro evaluation)
_SIMPLE_EXPR_RE = re.compile(
    r'^[\d\s+\-*/()xXa-fA-FuUlL]+$'
)

# Identifier extraction
_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')

# SystemVerilog keywords to exclude from identifier matching
_SV_KEYWORDS: Set[str] = {
    # Verilog keywords
    "always", "and", "assign", "automatic", "begin", "buf", "bufif0", "bufif1",
    "case", "casex", "casez", "cmos", "deassign", "default", "defparam", "disable",
    "edge", "else", "end", "endcase", "endfunction", "endgenerate", "endmodule",
    "endprimitive", "endspecify", "endtable", "endtask", "event", "for", "force",
    "forever", "fork", "function", "generate", "genvar", "highz0", "highz1", "if",
    "ifnone", "initial", "inout", "input", "integer", "join", "medium", "module",
    "nand", "negedge", "nmos", "nor", "noshowcancelled", "not", "notif0", "notif1",
    "or", "output", "parameter", "pmos", "posedge", "primitive", "pull0", "pull1",
    "pulldown", "pullup", "pulsestyle_onevent", "pulsestyle_ondetect", "rcmos",
    "real", "realtime", "reg", "release", "repeat", "rnmos", "rpmos", "rtran",
    "rtranif0", "rtranif1", "scalared", "showcancelled", "signed", "small",
    "specify", "specparam", "strong0", "strong1", "supply0", "supply1", "table",
    "task", "time", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1", "triand",
    "trior", "trireg", "unsigned", "uwire", "vectored", "wait", "wand", "weak0",
    "weak1", "while", "wire", "wor", "xnor", "xor",
    # SystemVerilog additions
    "accept_on", "alias", "always_comb", "always_ff", "always_latch", "assert",
    "assume", "attribute", "before", "bind", "bins", "binsof", "bit", "break",
    "byte", "chandle", "class", "clocking", "const", "constraint", "context",
    "continue", "cover", "covergroup", "coverpoint", "cross", "cut", "default",
    "dist", "do", "during", "else", "enum", "eventually", "exclude", "expect",
    "export", "extends", "extern", "final", "first_match", "foreach", "forever",
    "fork", "forkjoin", "function", "global", "iff", "ignore_bins", "illegal_bins",
    "implies", "import", "incdir", "include", "inside", "instance", "int",
    "interface", "intersect", "join", "join_any", "join_none", "local", "localparam",
    "logic", "longint", "matches", "modport", "nexttime", "null", "package",
    "packed", "parameter", "piecewise", "property", "protected", "public", "rand",
    "randc", "randcase", "randsequence", "ref", "reject_on", "release", "repeat",
    "restrict", "return", "sequence", "s_always", "s_eventually", "s_nexttime",
    "s_until", "s_until_with", "shortint", "shortreal", "signed", "soft",
    "solve", "static", "string", "struct", "super", "sync_accept_on", "sync_reject_on",
    "this", "throughout", "timeprecision", "timeunit", "type", "typedef", "union",
    "unique", "unsigned", "until", "until_with", "untyped", "var", "virtual",
    "void", "wait", "wait_order", "weak", "wildcard", "with", "within",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HeaderContextBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class HDLContextBuilder:
    """
    Resolves ``include directives and import packages, extracts SystemVerilog/Verilog
    type definitions from include files. Works entirely via regex — no external tooling required.

    Usage::

        builder = HDLContextBuilder("/path/to/hdl_project")
        includes = builder.resolve_includes("/path/to/hdl_project/src/top.sv")
        context  = builder.build_context_for_chunk(chunk_text, includes)
    """

    # Default directory names to skip during recursive include search
    _DEFAULT_WALK_EXCLUDE = {
        ".git", "build", "dist", "sim", "synth", "synthesis", "synthesis_results",
        "vendor", ".ccls-cache", "__pycache__", "bin", "obj", ".svn",
    }

    def __init__(
        self,
        codebase_path: str,
        include_paths: Optional[List[str]] = None,
        max_header_depth: int = 2,
        max_context_chars: int = 6000,
        exclude_system_headers: bool = True,
        max_definitions_per_header: int = 500,
        exclude_dirs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        exclude_headers: Optional[List[str]] = None,
    ):
        self.codebase_path = Path(codebase_path).resolve()
        self.include_paths: List[Path] = []
        for p in (include_paths or []):
            ip = Path(p)
            if not ip.is_absolute():
                ip = self.codebase_path / ip
            self.include_paths.append(ip.resolve())
        self.max_header_depth = max_header_depth
        self.max_context_chars = max_context_chars
        self.exclude_system_headers = exclude_system_headers
        self.max_definitions_per_header = max_definitions_per_header

        # Merge caller-supplied exclusions with defaults
        self.exclude_dirs = self._DEFAULT_WALK_EXCLUDE | set(exclude_dirs or [])
        self.exclude_globs = exclude_globs or []

        # User-specified headers to exclude (exact names, basenames, or glob patterns)
        self.exclude_headers: Set[str] = set(exclude_headers or [])

        # Caches (persist for the lifetime of this builder instance)
        self._include_cache: Dict[str, List[ResolvedInclude]] = {}
        self._header_cache: Dict[str, HeaderDefinitions] = {}

    # ─── Header Exclusion ─────────────────────────────────────────────────

    def _is_header_excluded(self, inc_name: str, resolved_path: Optional[str]) -> bool:
        """Check if a header matches the user-specified exclude list.

        Checks against: exact include name, basename of include name,
        basename of resolved path, and fnmatch glob patterns.
        """
        if not self.exclude_headers:
            return False
        basename_inc = os.path.basename(inc_name)
        basename_resolved = os.path.basename(resolved_path) if resolved_path else ""
        for pattern in self.exclude_headers:
            # Exact match on include name or basenames
            if pattern in (inc_name, basename_inc, basename_resolved):
                return True
            # Glob match
            if fnmatch.fnmatch(inc_name, pattern) or fnmatch.fnmatch(basename_inc, pattern):
                return True
            if resolved_path and fnmatch.fnmatch(resolved_path, pattern):
                return True
        return False

    # ─── Include Resolution ──────────────────────────────────────────────

    def resolve_includes(
        self,
        file_path: str,
        _depth: int = 0,
        _visited: Optional[Set[str]] = None,
    ) -> List[ResolvedInclude]:
        """
        Resolve ``include directives from *file_path* (and recursively
        from the resolved include files up to *max_header_depth*).

        Returns a flat, deduplicated list of :class:`ResolvedInclude`.
        """
        abs_path = str(Path(file_path).resolve())

        if abs_path in self._include_cache:
            return self._include_cache[abs_path]

        if _visited is None:
            _visited = set()

        if abs_path in _visited:
            return []  # circular include guard
        _visited.add(abs_path)

        result: List[ResolvedInclude] = []
        seen_paths: Set[str] = set()

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except (OSError, IOError) as exc:
            logger.debug("Cannot read %s for include resolution: %s", abs_path, exc)
            return []

        file_dir = Path(abs_path).parent

        for match in _INCLUDE_RE.finditer(content):
            inc_name = match.group(1)
            inc_type = "local"  # SystemVerilog `include is always local-style

            # Skip system packages if configured
            if self.exclude_system_headers:
                # Check against known standard packages
                basename = os.path.basename(inc_name).replace(".svh", "").replace(".vh", "")
                if inc_name in _SYSTEM_PACKAGES or basename in _SYSTEM_PACKAGES:
                    continue

        # Also process import statements
        for match in _IMPORT_RE.finditer(content):
            pkg_name = match.group(1)
            if self.exclude_system_headers and pkg_name in _SYSTEM_PACKAGES:
                continue
            # Note: imports don't have file paths, so we skip resolution for now

            resolved_path = self._resolve_include_path(inc_name, inc_type, file_dir)

            # Skip user-excluded headers
            if self._is_header_excluded(inc_name, resolved_path):
                logger.debug("  Skipping excluded header: %s", inc_name)
                continue

            ri = ResolvedInclude(
                name=inc_name,
                abs_path=resolved_path,
                include_type=inc_type,
                resolved=resolved_path is not None,
            )

            if resolved_path and resolved_path not in seen_paths:
                seen_paths.add(resolved_path)
                result.append(ri)
                logger.debug(
                    "  Resolved include: %s (%s) → %s", inc_name, inc_type, resolved_path
                )

                # Recurse into the resolved header
                if _depth < self.max_header_depth:
                    sub_includes = self.resolve_includes(
                        resolved_path, _depth=_depth + 1, _visited=_visited
                    )
                    for si in sub_includes:
                        if si.abs_path and si.abs_path not in seen_paths:
                            seen_paths.add(si.abs_path)
                            result.append(si)
            elif not resolved_path:
                # Unresolved — log diagnostic info to help debug
                logger.debug(
                    "  Unresolved include: %s (%s) in %s "
                    "(codebase_path=%s, file_dir=%s, include_paths=%s)",
                    inc_name, inc_type, abs_path,
                    self.codebase_path, file_dir,
                    [str(p) for p in self.include_paths],
                )

        self._include_cache[abs_path] = result
        return result

    def _resolve_include_path(
        self, inc_name: str, inc_type: str, file_dir: Path
    ) -> Optional[str]:
        """Try to find the actual file for an `include directive.

        SystemVerilog `include is local-style. We search the current file directory,
        include_paths, codebase root, and common HDL include subdirectories.
        """
        search_dirs: List[Path] = []

        # Local includes (SystemVerilog `include is always local-style)
        # Same dir → include_paths → codebase root
        search_dirs = [file_dir] + self.include_paths + [self.codebase_path]

        # Also add common HDL include subdirectories
        for subdir in ("include", "inc", "src", "rtl", "hdl", "common", "interfaces", "packages"):
            candidate_dir = self.codebase_path / subdir
            if candidate_dir.is_dir() and candidate_dir not in search_dirs:
                search_dirs.append(candidate_dir)

        for search_dir in search_dirs:
            candidate = search_dir / inc_name
            if candidate.is_file():
                return str(candidate.resolve())

        # Fallback: recursive search under codebase_path for the basename
        # (handles cases like #include "subdir/header.h" when cwd is wrong,
        #  or headers in deeply nested project directories)
        basename = os.path.basename(inc_name)
        for root, _dirs, files in os.walk(self.codebase_path):
            # Apply directory-name exclusions
            _dirs[:] = [d for d in _dirs if d not in self.exclude_dirs]

            if basename in files:
                candidate = Path(root) / basename
                # Apply glob exclusions on the candidate
                if self.exclude_globs:
                    try:
                        rel = candidate.relative_to(self.codebase_path).as_posix().lower()
                        if any(fnmatch.fnmatch(rel, g.lower()) for g in self.exclude_globs):
                            continue
                    except ValueError:
                        pass
                # Verify the relative path matches if inc_name has directories
                if "/" in inc_name or "\\" in inc_name:
                    try:
                        candidate_rel = candidate.relative_to(self.codebase_path).as_posix()
                        if candidate_rel.endswith(inc_name.replace("\\", "/")):
                            return str(candidate.resolve())
                    except ValueError:
                        pass
                else:
                    return str(candidate.resolve())

        return None

    # ─── Header Parsing ──────────────────────────────────────────────────

    def parse_include(self, file_path: str) -> IncludeDefinitions:
        """
        Parse an include file and extract enum, struct, macro, typedef,
        parameter, port, module, interface, task, and function definitions.

        Results are cached by absolute path.
        """
        abs_path = str(Path(file_path).resolve())

        if abs_path in self._header_cache:
            return self._header_cache[abs_path]

        defs = IncludeDefinitions(file_path=abs_path)

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except (OSError, IOError) as exc:
            logger.debug("Cannot read header %s: %s", abs_path, exc)
            self._header_cache[abs_path] = defs
            return defs

        # Strip single-line comments for cleaner parsing
        # (preserve line structure for multi-line constructs)
        cleaned = re.sub(r'//[^\n]*', '', content)
        # Strip block comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

        self._parse_enums(cleaned, defs)
        self._parse_structs(cleaned, defs)
        self._parse_macros(content, defs)  # Use original for macros (` directives)
        self._parse_typedefs(cleaned, defs)
        self._parse_parameters(cleaned, defs)
        self._parse_module_decls(cleaned, defs)
        self._parse_interface_decls(cleaned, defs)
        self._parse_task_protos(cleaned, defs)
        self._parse_function_protos(cleaned, defs)

        # Enforce per-include definition limit
        total = (len(defs.enums) + len(defs.structs) + len(defs.macros)
                 + len(defs.typedefs) + len(defs.parameters) + len(defs.module_decls)
                 + len(defs.interface_decls) + len(defs.task_protos) + len(defs.function_protos))
        if total > self.max_definitions_per_header:
            logger.debug("Include file %s has %d definitions, truncating to %d",
                         abs_path, total, self.max_definitions_per_header)

        self._header_cache[abs_path] = defs
        return defs

    def _parse_enums(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract enum definitions (both plain and typedef'd)."""
        # typedef enum [tag] { ... } alias;
        for m in _ENUM_TYPEDEF_RE.finditer(content):
            tag = m.group(1) or ""
            body = m.group(2)
            alias = m.group(3)
            name = alias or tag or "_anon_enum"
            members = self._parse_enum_members(body)
            raw = self._format_enum(name, members)
            defs.enums.append(EnumDef(
                name=name, members=members, is_typedef=True, raw=raw
            ))

        # Plain enum name { ... }
        for m in _ENUM_PLAIN_RE.finditer(content):
            name = m.group(1)
            body = m.group(2)
            # Skip if already captured by typedef variant
            if any(e.name == name for e in defs.enums):
                continue
            members = self._parse_enum_members(body)
            raw = self._format_enum(name, members)
            defs.enums.append(EnumDef(
                name=name, members=members, is_typedef=False, raw=raw
            ))

    def _parse_enum_members(self, body: str) -> List[EnumMember]:
        """Parse enum member list, tracking auto-increment values."""
        members: List[EnumMember] = []
        auto_val = 0
        for line in body.split(","):
            line = line.strip()
            if not line:
                continue
            # Remove trailing comments
            line = re.sub(r'/\*.*?\*/', '', line).strip()
            line = re.sub(r'//.*$', '', line).strip()
            if not line:
                continue

            if "=" in line:
                parts = line.split("=", 1)
                name = parts[0].strip()
                val_str = parts[1].strip()
                num_val = self._try_parse_int(val_str)
                members.append(EnumMember(name=name, value=val_str, numeric_value=num_val))
                if num_val is not None:
                    auto_val = num_val + 1
                else:
                    auto_val += 1
            else:
                name = line.strip()
                if not re.match(r'^[A-Za-z_]\w*$', name):
                    continue
                members.append(EnumMember(
                    name=name, value=None, numeric_value=auto_val
                ))
                auto_val += 1
        return members

    @staticmethod
    def _format_enum(name: str, members: List[EnumMember]) -> str:
        """Format an enum as a single-line C declaration."""
        parts = []
        for m in members:
            if m.value is not None:
                parts.append(f"{m.name} = {m.value}")
            elif m.numeric_value is not None:
                parts.append(f"{m.name} = {m.numeric_value}")
            else:
                parts.append(m.name)
        return f"enum {name} {{ {', '.join(parts)} }};"

    def _parse_structs(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract struct/union definitions."""
        # typedef struct/union [tag] { ... } alias;
        for m in _STRUCT_TYPEDEF_RE.finditer(content):
            kind = m.group(1)  # struct or union
            tag = m.group(2) or ""
            body_start = m.end() - 1  # position of '{'
            body, body_end = self._match_braces(content, body_start)
            if body is None:
                continue
            # Look for alias after closing brace
            after = content[body_end:body_end + 100].strip()
            alias_m = re.match(r'(\w+)\s*;', after)
            alias = alias_m.group(1) if alias_m else ""
            name = alias or tag or f"_anon_{kind}"
            fields = self._parse_struct_fields(body)
            raw = self._format_struct(kind, name, fields)
            defs.structs.append(StructDef(
                name=name, kind=kind, fields=fields, is_typedef=True, raw=raw
            ))

        # Plain struct/union name { ... }
        for m in _STRUCT_PLAIN_RE.finditer(content):
            kind = m.group(1)
            name = m.group(2)
            if any(s.name == name for s in defs.structs):
                continue
            body_start = m.end() - 1
            body, _ = self._match_braces(content, body_start)
            if body is None:
                continue
            fields = self._parse_struct_fields(body)
            raw = self._format_struct(kind, name, fields)
            defs.structs.append(StructDef(
                name=name, kind=kind, fields=fields, is_typedef=False, raw=raw
            ))

    @staticmethod
    def _match_braces(content: str, start: int) -> Tuple[Optional[str], int]:
        """Find matching closing brace, returns (body_text, end_pos)."""
        if start >= len(content) or content[start] != '{':
            return None, start
        depth = 0
        i = start
        while i < len(content):
            ch = content[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return content[start + 1:i], i + 1
            elif ch == '"' or ch == "'":
                # Skip string/char literals
                quote = ch
                i += 1
                while i < len(content) and content[i] != quote:
                    if content[i] == '\\':
                        i += 1
                    i += 1
            i += 1
        return None, start

    def _parse_struct_fields(self, body: str) -> List[StructField]:
        """Parse struct body to extract field declarations."""
        fields: List[StructField] = []
        # Remove nested structs/unions/enums for simpler field parsing
        cleaned = re.sub(r'(struct|union|enum)\s*\w*\s*\{[^}]*\}\s*\w*\s*;', '', body, flags=re.DOTALL)

        for line in cleaned.split(";"):
            line = line.strip()
            if not line:
                continue
            # Remove __attribute__, __aligned, etc.
            line = re.sub(r'__\w+__\s*\([^)]*\)', '', line)
            line = re.sub(r'__\w+', '', line)
            line = line.strip()
            if not line:
                continue

            # Check for array
            arr_match = re.match(r'(.*?)\s+(\w+)\s*\[([\w\s+\-*/]*)\]$', line)
            if arr_match:
                type_name = arr_match.group(1).strip()
                field_name = arr_match.group(2).strip()
                arr_size = arr_match.group(3).strip() or None
                if type_name and field_name and re.match(r'^[A-Za-z_]', field_name):
                    fields.append(StructField(type_name, field_name, arr_size))
                continue

            # Regular field: type name;
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                type_name = parts[0].strip()
                field_name = parts[1].strip().rstrip("*")
                if re.match(r'^\*?[A-Za-z_]\w*$', field_name):
                    # Handle pointer in field name
                    if parts[1].strip().startswith("*"):
                        type_name += " *"
                        field_name = parts[1].strip().lstrip("*")
                    fields.append(StructField(type_name, field_name))
        return fields

    @staticmethod
    def _format_struct(kind: str, name: str, fields: List[StructField]) -> str:
        """Format struct/union as a compact C declaration."""
        parts = []
        for f in fields:
            if f.array_size:
                parts.append(f"{f.type_name} {f.field_name}[{f.array_size}]")
            else:
                parts.append(f"{f.type_name} {f.field_name}")
        return f"{kind} {name} {{ {'; '.join(parts)}; }};"

    def _parse_macros(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract `define macros, handling multi-line continuations."""
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            # Collect continuation lines
            full_line = line
            while _MACRO_CONTINUATION_RE.search(full_line) and i + 1 < len(lines):
                i += 1
                full_line = full_line.rstrip().rstrip('\\') + ' ' + lines[i].strip()

            m = _MACRO_RE.match(full_line.strip())
            if m:
                name = m.group(1)
                params = m.group(2)  # None if not function-like
                value = m.group(3).strip()

                # Skip include guards
                if _INCLUDE_GUARD_RE.match(name):
                    i += 1
                    continue

                # Skip empty defines
                if not value:
                    i += 1
                    continue

                is_func = params is not None
                is_numeric, num_val = self._classify_macro_value(value)

                raw = f"`define {name}"
                if is_func:
                    raw += f"({params})"
                raw += f" {value}"

                defs.macros.append(MacroDef(
                    name=name, value=value, is_numeric=is_numeric,
                    numeric_value=num_val, is_function_like=is_func, raw=raw
                ))

            i += 1

    def _parse_typedefs(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract simple typedefs (not enum/struct which are handled separately)."""
        for m in _TYPEDEF_SIMPLE_RE.finditer(content):
            original = m.group(1).strip()
            alias = m.group(2).strip()
            if not alias or not re.match(r'^[A-Za-z_]\w*$', alias.lstrip('*')):
                continue
            raw = f"typedef {original} {alias};"
            defs.typedefs.append(TypedefDef(alias=alias.lstrip('*'), original_type=original, raw=raw))

    def _parse_function_protos(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract function/task declarations (SystemVerilog functions)."""
        for m in _FUNCTION_PROTO_RE.finditer(content):
            ret_type = m.group(1).strip()
            func_name = m.group(2).strip()
            params = m.group(3).strip()

            # Skip if name is a SystemVerilog keyword
            if func_name in _SV_KEYWORDS:
                continue

            raw = f"function {ret_type} {func_name}({params});"
            defs.function_protos.append(FunctionProto(
                name=func_name, return_type=ret_type, params=params, raw=raw
            ))

    def _parse_parameters(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract parameter and localparam declarations."""
        for m in _PARAMETER_RE.finditer(content):
            param_type = m.group(1)  # "parameter" or "localparam"
            data_type = m.group(2).strip()
            name = m.group(3).strip()
            default_value = m.group(4).strip() if m.group(4) else None

            if not re.match(r'^[A-Za-z_]\w*$', name):
                continue

            is_localparam = param_type == "localparam"
            raw = f"{param_type} {data_type} {name}"
            if default_value:
                raw += f" = {default_value}"
            raw += ";"

            defs.parameters.append(ParameterDef(
                name=name, data_type=data_type, default_value=default_value,
                is_localparam=is_localparam, raw=raw
            ))

    def _parse_module_decls(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract module declarations."""
        for m in _MODULE_DECL_RE.finditer(content):
            module_name = m.group(1)
            params_str = m.group(2) or ""
            ports_str = m.group(3) or ""

            if not module_name or not re.match(r'^[A-Za-z_]\w*$', module_name):
                continue

            params = [p.strip() for p in params_str.split(',') if p.strip()]
            ports = [p.strip() for p in ports_str.split(',') if p.strip()]

            raw = f"module {module_name}"
            if params:
                raw += f" #({params_str})"
            raw += f" ({ports_str});"

            defs.module_decls.append(ModuleDecl(
                name=module_name, ports=ports, parameters=params, raw=raw
            ))

    def _parse_interface_decls(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract interface declarations."""
        for m in _INTERFACE_DECL_RE.finditer(content):
            interface_name = m.group(1)
            params_str = m.group(2) or ""

            if not interface_name or not re.match(r'^[A-Za-z_]\w*$', interface_name):
                continue

            params = [p.strip() for p in params_str.split(',') if p.strip()]

            raw = f"interface {interface_name}"
            if params:
                raw += f" #({params_str})"
            raw += ";"

            defs.interface_decls.append(InterfaceDecl(
                name=interface_name, parameters=params, raw=raw
            ))

    def _parse_task_protos(self, content: str, defs: IncludeDefinitions) -> None:
        """Extract task declarations."""
        for m in _TASK_PROTO_RE.finditer(content):
            return_type = m.group(1) or "void"
            task_name = m.group(2)
            params = m.group(3) or ""

            if not task_name or not re.match(r'^[A-Za-z_]\w*$', task_name):
                continue

            raw = f"task {task_name}({params});"

            defs.task_protos.append(TaskProto(
                name=task_name, params=params, raw=raw
            ))

    # ─── Value Classification ────────────────────────────────────────────

    @staticmethod
    def _try_parse_int(val_str: str) -> Optional[int]:
        """Try to evaluate a string as an integer constant."""
        val = val_str.strip().rstrip("uUlL")
        try:
            return int(val, 0)
        except (ValueError, TypeError):
            pass
        # Try simple arithmetic
        if _SIMPLE_EXPR_RE.match(val):
            clean = re.sub(r'[uUlL]', '', val)
            try:
                return int(eval(clean))  # nosec - only numeric expressions
            except Exception:
                pass
        return None

    def _classify_macro_value(self, value: str) -> Tuple[bool, Optional[int]]:
        """Classify a macro value as numeric or not."""
        num = self._try_parse_int(value)
        if num is not None:
            return True, num
        # Check for string literal
        if value.startswith('"') or value.startswith("'"):
            return False, None
        # Check for simple sizeof expression
        if value.startswith("sizeof"):
            return False, None
        # Check for cast expression wrapping a number
        cast_m = re.match(r'\(\s*[\w\s*]+\s*\)\s*([\dxXa-fA-F]+[uUlL]*)', value)
        if cast_m:
            num = self._try_parse_int(cast_m.group(1))
            return num is not None, num
        return False, None

    # ─── Context Building ────────────────────────────────────────────────

    def build_context_for_chunk(
        self,
        chunk_text: str,
        file_includes: List[ResolvedInclude],
    ) -> str:
        """
        Build a concise include context string containing only definitions
        that are actually referenced in *chunk_text*.

        Respects ``max_context_chars`` budget with priority ordering:
        parameters > macros > enums > structs > typedefs > module/interface decls > function protos.
        """
        if not file_includes:
            return ""

        # Extract identifiers from chunk (minus keywords)
        chunk_idents = set(_IDENT_RE.findall(chunk_text)) - _SV_KEYWORDS

        if not chunk_idents:
            return ""

        # Collect all relevant definitions across all resolved includes
        relevant_parameters: List[str] = []
        relevant_macros: List[str] = []
        relevant_enums: List[str] = []
        relevant_structs: List[str] = []
        relevant_typedefs: List[str] = []
        relevant_modules: List[str] = []
        relevant_interfaces: List[str] = []
        relevant_tasks: List[str] = []
        relevant_functions: List[str] = []

        for inc in file_includes:
            if not inc.resolved or not inc.abs_path:
                continue

            hdefs = self.parse_include(inc.abs_path)

            # Parameters: include if parameter name is referenced
            for pdef in hdefs.parameters:
                if pdef.name in chunk_idents:
                    relevant_parameters.append(pdef.raw)

            # Macros: include if macro name is referenced
            for mdef in hdefs.macros:
                if mdef.name in chunk_idents:
                    relevant_macros.append(mdef.raw)

            # Enums: include if enum name or any member name is referenced
            for edef in hdefs.enums:
                if edef.name in chunk_idents:
                    relevant_enums.append(edef.raw)
                    continue
                # Check if any member is referenced
                for mem in edef.members:
                    if mem.name in chunk_idents:
                        relevant_enums.append(edef.raw)
                        break

            # Structs: include if struct name is referenced
            for sdef in hdefs.structs:
                if sdef.name in chunk_idents:
                    relevant_structs.append(sdef.raw)

            # Typedefs: include if alias is referenced
            for tdef in hdefs.typedefs:
                if tdef.alias in chunk_idents:
                    relevant_typedefs.append(tdef.raw)

            # Module declarations: include if module name is referenced
            for mdef in hdefs.module_decls:
                if mdef.name in chunk_idents:
                    relevant_modules.append(mdef.raw)

            # Interface declarations: include if interface name is referenced
            for idef in hdefs.interface_decls:
                if idef.name in chunk_idents:
                    relevant_interfaces.append(idef.raw)

            # Task prototypes: include if task name is referenced
            for tdef in hdefs.task_protos:
                if tdef.name in chunk_idents:
                    relevant_tasks.append(tdef.raw)

            # Function prototypes: include if function name is referenced
            for fproto in hdefs.function_protos:
                if fproto.name in chunk_idents:
                    relevant_functions.append(fproto.raw)

        # Deduplicate
        relevant_parameters = list(dict.fromkeys(relevant_parameters))
        relevant_macros = list(dict.fromkeys(relevant_macros))
        relevant_enums = list(dict.fromkeys(relevant_enums))
        relevant_structs = list(dict.fromkeys(relevant_structs))
        relevant_typedefs = list(dict.fromkeys(relevant_typedefs))
        relevant_modules = list(dict.fromkeys(relevant_modules))
        relevant_interfaces = list(dict.fromkeys(relevant_interfaces))
        relevant_tasks = list(dict.fromkeys(relevant_tasks))
        relevant_functions = list(dict.fromkeys(relevant_functions))

        # Nothing relevant found
        if not any([relevant_parameters, relevant_macros, relevant_enums,
                     relevant_structs, relevant_typedefs, relevant_modules,
                     relevant_interfaces, relevant_tasks, relevant_functions]):
            return ""

        # Build context string respecting token budget (priority order)
        sections: List[Tuple[str, List[str]]] = [
            ("Parameters", relevant_parameters),
            ("Macros", relevant_macros),
            ("Enums", relevant_enums),
            ("Structs", relevant_structs),
            ("Typedefs", relevant_typedefs),
            ("Module declarations", relevant_modules),
            ("Interface declarations", relevant_interfaces),
            ("Task declarations", relevant_tasks),
            ("Function declarations", relevant_functions),
        ]

        header_line = "// ──── INCLUDE CONTEXT (from included .svh/.vh files) ────"
        footer_line = "// ──── END INCLUDE CONTEXT ────"
        budget = self.max_context_chars - len(header_line) - len(footer_line) - 20

        parts: List[str] = []
        used = 0

        for section_name, items in sections:
            if not items or used >= budget:
                break
            section_header = f"// {section_name}:"
            section_lines = [section_header]
            for item in items:
                line_len = len(item) + 1  # +1 for newline
                if used + line_len > budget:
                    break
                section_lines.append(item)
                used += line_len
            if len(section_lines) > 1:  # Has items beyond the header
                parts.append("\n".join(section_lines))
                used += len(section_header) + 2  # header + spacing

        if not parts:
            return ""

        return f"{header_line}\n" + "\n\n".join(parts) + f"\n{footer_line}"

    # ─── Convenience ─────────────────────────────────────────────────────

    def get_file_context(self, file_path: str) -> str:
        """
        Get the full include context for a file (all definitions from all
        resolved includes, not filtered by chunk content).

        Useful for small files that are a single chunk.
        """
        includes = self.resolve_includes(file_path)
        if not includes:
            return ""

        all_raws: List[str] = []
        for inc in includes:
            if not inc.resolved or not inc.abs_path:
                continue
            hdefs = self.parse_include(inc.abs_path)
            for pdef in hdefs.parameters:
                all_raws.append(pdef.raw)
            for mdef in hdefs.macros:
                all_raws.append(mdef.raw)
            for edef in hdefs.enums:
                all_raws.append(edef.raw)
            for sdef in hdefs.structs:
                all_raws.append(sdef.raw)
            for tdef in hdefs.typedefs:
                all_raws.append(tdef.raw)
            for mdef in hdefs.module_decls:
                all_raws.append(mdef.raw)
            for idef in hdefs.interface_decls:
                all_raws.append(idef.raw)
            for tdef in hdefs.task_protos:
                all_raws.append(tdef.raw)
            for fproto in hdefs.function_protos:
                all_raws.append(fproto.raw)

        if not all_raws:
            return ""

        # Truncate to budget
        header_line = "// ──── INCLUDE CONTEXT (from included .svh/.vh files) ────"
        footer_line = "// ──── END INCLUDE CONTEXT ────"
        result_lines = [header_line]
        used = len(header_line) + len(footer_line) + 10
        for raw in all_raws:
            if used + len(raw) + 1 > self.max_context_chars:
                break
            result_lines.append(raw)
            used += len(raw) + 1
        result_lines.append(footer_line)
        return "\n".join(result_lines)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for debugging/telemetry."""
        total_defs = 0
        for hdefs in self._header_cache.values():
            total_defs += (len(hdefs.enums) + len(hdefs.structs) + len(hdefs.macros)
                           + len(hdefs.typedefs) + len(hdefs.function_protos)
                           + len(hdefs.extern_vars))
        return {
            "headers_parsed": len(self._header_cache),
            "includes_resolved": len(self._include_cache),
            "total_definitions_cached": total_defs,
        }
