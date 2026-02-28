"""
PDF Dependency Graph Report Generator for CARE.

Produces a professional, multi-page PDF report from HDL dependency analysis
using ReportLab, featuring module hierarchy visualization, include dependencies,
package imports, parameter overrides, interface bindings, and comprehensive
issue analysis with CARE branding.

Usage:
    from agents.reports.dependency_report_pdf import DependencyGraphPDFReport
    from agents.services import DependencyGraph

    graph = DependencyGraph(...)  # from dependency analysis
    report = DependencyGraphPDFReport(graph, config={"project_name": "MyProject"})
    report.generate("dependency_report.pdf")
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agents.services import DependencyGraph

logger = logging.getLogger("reports.dependency_pdf")

# ── CARE Branding Palette ──────────────────────────────────────────────
NAVY = colors.HexColor("#0B1929")
TEAL = colors.HexColor("#00D4AA")
BLUE = colors.HexColor("#0EA5E9")
PURPLE = colors.HexColor("#8B5CF6")
ORANGE = colors.HexColor("#F59E0B")
RED = colors.HexColor("#EF4444")
GREEN = colors.HexColor("#22C55E")
DARK_GRAY = colors.HexColor("#333333")
MED_GRAY = colors.HexColor("#666666")
LIGHT_GRAY = colors.HexColor("#F5F5F5")
WHITE = colors.white
HEADER_BG = colors.HexColor("#0B1929")
ROW_ALT = colors.HexColor("#F0F9FF")


# ── Custom Styles ──────────────────────────────────────────────────────
def _build_styles():
    """Build custom ReportLab paragraph styles."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "CareTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=NAVY,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "CareSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            textColor=MED_GRAY,
            spaceAfter=18,
        ),
        "heading1": ParagraphStyle(
            "CareH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=NAVY,
            spaceBefore=20,
            spaceAfter=10,
        ),
        "heading2": ParagraphStyle(
            "CareH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "CareBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_GRAY,
            leading=14,
            spaceAfter=6,
        ),
        "body_bold": ParagraphStyle(
            "CareBodyBold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=DARK_GRAY,
            leading=14,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "CareCell",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_GRAY,
            leading=11,
        ),
        "cell_header": ParagraphStyle(
            "CareCellHeader",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=WHITE,
            leading=11,
        ),
        "cell_bold": ParagraphStyle(
            "CareCellBold",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=DARK_GRAY,
            leading=11,
        ),
        "metric_value": ParagraphStyle(
            "MetricVal",
            fontName="Helvetica-Bold",
            fontSize=24,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName="Helvetica",
            fontSize=9,
            textColor=MED_GRAY,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "CareFooter",
            fontName="Helvetica",
            fontSize=8,
            textColor=MED_GRAY,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ── Severity Color Mapping ────────────────────────────────────────────
def _severity_color(severity: str) -> colors.HexColor:
    """Map severity to color."""
    severity_map = {
        "critical": RED,
        "high": ORANGE,
        "medium": colors.HexColor("#F59E0B"),
        "low": GREEN,
    }
    return severity_map.get(severity.lower(), MED_GRAY)


# ── Report Class ───────────────────────────────────────────────────────
class DependencyGraphPDFReport:
    """
    Generates a professional PDF dependency graph report from HDL analysis.

    Args:
        graph: DependencyGraph object from HDLDependencyAnalyzer.analyze()
        config: Optional config dict with keys:
            - project_name: Name of the project (default: "HDL Design")
            - output_dir: Output directory (default: current directory)
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph = graph
        self.config = config or {}
        self.styles = _build_styles()
        self.story: List[Any] = []

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, output_path: str) -> str:
        """
        Generate the PDF report and write to output_path.

        Args:
            output_path: Path where the PDF should be written.

        Returns:
            The output path (for convenience).
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        self._build_title_section()
        self._build_executive_summary()
        self._build_module_hierarchy_section()
        self._build_include_map_section()
        self._build_package_section()
        self._build_parameter_section()
        self._build_issues_section()
        self._build_methodology_section()

        doc.build(
            self.story,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer,
        )
        logger.info(f"Dependency report written to {output_path}")
        return output_path

    # ── Header / Footer ────────────────────────────────────────────────

    def _header_footer(self, canvas, doc):
        """Draw header and footer on every page."""
        canvas.saveState()
        w, h = letter

        # Header line (teal accent)
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(2)
        canvas.line(0.75 * inch, h - 0.6 * inch, w - 0.75 * inch, h - 0.6 * inch)

        # Header text
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(0.75 * inch, h - 0.55 * inch, "CARE")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MED_GRAY)
        canvas.drawString(
            1.15 * inch, h - 0.55 * inch, "  |  HDL Dependency Analysis Report"
        )

        # Footer
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MED_GRAY)
        canvas.drawCentredString(w / 2, 0.45 * inch, f"Page {doc.page}")
        date_str = datetime.now().strftime("%B %d, %Y")
        canvas.drawRightString(w - 0.75 * inch, 0.45 * inch, date_str)

        canvas.restoreState()

    # ── Sections ───────────────────────────────────────────────────────

    def _build_title_section(self):
        """Build title/cover section."""
        s = self.styles
        project = self.config.get("project_name", "HDL Design")

        self.story.append(Spacer(1, 0.3 * inch))
        self.story.append(Paragraph("HDL Dependency Analysis Report", s["title"]))
        self.story.append(
            Paragraph(
                f"Project: {project}  |  Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
                s["subtitle"],
            )
        )
        self.story.append(
            Paragraph(
                f"Files Analyzed: <b>{self.graph.metadata.files_analyzed}</b>",
                s["body"],
            )
        )
        self.story.append(Spacer(1, 0.2 * inch))

    def _build_executive_summary(self):
        """Build executive summary with key statistics."""
        s = self.styles
        summary = self.graph.score_summary

        self.story.append(Paragraph("1. Executive Summary", s["heading1"]))

        # Key stats in 2x4 table
        stat_rows = [
            [
                Paragraph("Modules", s["cell_header"]),
                Paragraph("Instantiations", s["cell_header"]),
                Paragraph("Max Depth", s["cell_header"]),
                Paragraph("Cycles", s["cell_header"]),
            ],
            [
                Paragraph(str(summary.get("modules", 0)), s["metric_value"]),
                Paragraph(str(summary.get("instantiations", 0)), s["metric_value"]),
                Paragraph(str(summary.get("max_depth", 0)), s["metric_value"]),
                Paragraph(str(summary.get("cycles", 0)), s["metric_value"]),
            ],
            [
                Paragraph("Includes", s["cell_header"]),
                Paragraph("Packages", s["cell_header"]),
                Paragraph("Interfaces", s["cell_header"]),
                Paragraph("Symbol Collisions", s["cell_header"]),
            ],
            [
                Paragraph(str(summary.get("includes", 0)), s["metric_value"]),
                Paragraph(str(summary.get("packages", 0)), s["metric_value"]),
                Paragraph(str(summary.get("interfaces", 0)), s["metric_value"]),
                Paragraph(str(summary.get("symbol_collisions", 0)), s["metric_value"]),
            ],
        ]

        tbl = Table(stat_rows, colWidths=[1.75 * inch] * 4)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                    ("BACKGROUND", (0, 2), (-1, 2), HEADER_BG),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("TEXTCOLOR", (0, 2), (-1, 2), WHITE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ]
            )
        )
        self.story.append(tbl)
        self.story.append(Spacer(1, 12))

    def _build_module_hierarchy_section(self):
        """Build module hierarchy section with top modules by fan-out."""
        s = self.styles
        self.story.append(Paragraph("2. Module Hierarchy", s["heading1"]))

        # Top 20 modules by fan-out
        modules_by_fanout = sorted(
            self.graph.module_hierarchy.modules.values(),
            key=lambda m: -m.fan_out,
        )[:20]

        if not modules_by_fanout:
            self.story.append(Paragraph("No modules to display.", s["body"]))
            return

        header = [
            Paragraph("Module", s["cell_header"]),
            Paragraph("Fan-In", s["cell_header"]),
            Paragraph("Fan-Out", s["cell_header"]),
            Paragraph("Depth", s["cell_header"]),
            Paragraph("File", s["cell_header"]),
        ]
        rows = [header]

        for module in modules_by_fanout:
            rows.append(
                [
                    Paragraph(module.name, s["cell_bold"]),
                    Paragraph(str(module.fan_in), s["cell"]),
                    Paragraph(str(module.fan_out), s["cell"]),
                    Paragraph(
                        self._get_module_depth(module.name) or "N/A", s["cell"]
                    ),
                    Paragraph(module.file_path, s["cell"]),
                ]
            )

        col_widths = [1.2 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 2.4 * inch]
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

        tbl.setStyle(TableStyle(style_cmds))
        self.story.append(tbl)
        self.story.append(Spacer(1, 12))

        # Root modules
        if self.graph.module_hierarchy.root_modules:
            self.story.append(Paragraph("Root Modules", s["heading2"]))
            roots_str = ", ".join(sorted(self.graph.module_hierarchy.root_modules))
            self.story.append(Paragraph(roots_str, s["body"]))
            self.story.append(Spacer(1, 8))

        # Cycle warnings
        if self.graph.module_hierarchy.cycles:
            self.story.append(Paragraph("Module Cycles Detected", s["heading2"]))
            for cycle in self.graph.module_hierarchy.cycles:
                cycle_str = " → ".join(cycle)
                self.story.append(
                    Paragraph(f"<b>Cycle:</b> {cycle_str}", s["body"])
                )
            self.story.append(Spacer(1, 8))

    def _build_include_map_section(self):
        """Build include dependency section."""
        s = self.styles
        self.story.append(Paragraph("3. Include Dependencies", s["heading1"]))

        # Files with most includes
        files_by_include_count = sorted(
            self.graph.include_tree.includes_by_file.items(),
            key=lambda x: -len(x[1]),
        )[:15]

        if files_by_include_count:
            header = [
                Paragraph("File", s["cell_header"]),
                Paragraph("Direct Includes", s["cell_header"]),
                Paragraph("Transitive Includes", s["cell_header"]),
            ]
            rows = [header]

            for file_path, includes in files_by_include_count:
                transitive = len(
                    self.graph.include_tree.include_chains.get(file_path, [])
                )
                rows.append(
                    [
                        Paragraph(file_path, s["cell_bold"]),
                        Paragraph(str(len(includes)), s["cell"]),
                        Paragraph(str(transitive), s["cell"]),
                    ]
                )

            col_widths = [3 * inch, 1.5 * inch, 1.5 * inch]
            tbl = Table(rows, colWidths=col_widths, repeatRows=1)

            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ]
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

            tbl.setStyle(TableStyle(style_cmds))
            self.story.append(tbl)
            self.story.append(Spacer(1, 12))

        # Unresolved includes
        if self.graph.include_tree.unresolved_includes:
            self.story.append(Paragraph("Unresolved Includes", s["heading2"]))
            for unresolved in self.graph.include_tree.unresolved_includes[:10]:
                self.story.append(
                    Paragraph(
                        f"<b>{unresolved.source_file}:{unresolved.line}</b> → "
                        f"<font color=\"red\">{unresolved.include_name}</font>",
                        s["body"],
                    )
                )
            self.story.append(Spacer(1, 8))

        # Circular includes
        if self.graph.include_tree.circular_includes:
            self.story.append(Paragraph("Circular Includes", s["heading2"]))
            for src, dst in self.graph.include_tree.circular_includes:
                self.story.append(
                    Paragraph(f"<b>{src}</b> ↔ <b>{dst}</b>", s["body"])
                )
            self.story.append(Spacer(1, 8))

    def _build_package_section(self):
        """Build package import section."""
        s = self.styles
        self.story.append(Paragraph("4. Package Imports", s["heading1"]))

        packages = self.graph.package_imports.package_defs
        if not packages:
            self.story.append(Paragraph("No packages defined.", s["body"]))
            return

        header = [
            Paragraph("Package", s["cell_header"]),
            Paragraph("Symbols Exported", s["cell_header"]),
            Paragraph("Files Importing", s["cell_header"]),
        ]
        rows = [header]

        for pkg_name, pkg_def in sorted(packages.items())[:15]:
            importing_count = sum(
                1
                for import_list in self.graph.package_imports.imports_by_file.values()
                for imp in import_list
                if imp.package_name == pkg_name
            )
            rows.append(
                [
                    Paragraph(pkg_name, s["cell_bold"]),
                    Paragraph(str(len(pkg_def.exported_symbols)), s["cell"]),
                    Paragraph(str(importing_count), s["cell"]),
                ]
            )

        col_widths = [2.5 * inch, 2 * inch, 2 * inch]
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

        tbl.setStyle(TableStyle(style_cmds))
        self.story.append(tbl)
        self.story.append(Spacer(1, 12))

        # Unresolved package imports
        if self.graph.package_imports.unresolved_packages:
            self.story.append(Paragraph("Unresolved Package Imports", s["heading2"]))
            for pkg in self.graph.package_imports.unresolved_packages:
                self.story.append(Paragraph(f"<font color=\"red\">{pkg}</font>", s["body"]))
            self.story.append(Spacer(1, 8))

    def _build_parameter_section(self):
        """Build parameter override section."""
        s = self.styles
        self.story.append(Paragraph("5. Parameter Overrides", s["heading1"]))

        overrides = self.graph.parameter_map.overrides
        if not overrides:
            self.story.append(Paragraph("No parameter overrides.", s["body"]))
            return

        header = [
            Paragraph("Instance", s["cell_header"]),
            Paragraph("Module", s["cell_header"]),
            Paragraph("Parameter", s["cell_header"]),
            Paragraph("Value", s["cell_header"]),
            Paragraph("Default", s["cell_header"]),
        ]
        rows = [header]

        for override in overrides[:15]:
            is_mismatch = override.type_mismatch
            rows.append(
                [
                    Paragraph(override.instance_name, s["cell_bold"]),
                    Paragraph(override.child_module, s["cell"]),
                    Paragraph(override.param_name, s["cell"]),
                    Paragraph(
                        f'<font color="{"red" if is_mismatch else "black"}">'
                        f"{override.override_value}</font>",
                        s["cell"],
                    ),
                    Paragraph(override.default_value, s["cell"]),
                ]
            )

        col_widths = [1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

        tbl.setStyle(TableStyle(style_cmds))
        self.story.append(tbl)
        self.story.append(Spacer(1, 12))

        # Type mismatch warnings
        if self.graph.parameter_map.type_mismatches:
            self.story.append(Paragraph("Parameter Type Mismatches", s["heading2"]))
            for mismatch in self.graph.parameter_map.type_mismatches[:10]:
                self.story.append(
                    Paragraph(
                        f"<b>{mismatch.instance_name}.</b>{mismatch.param_name}: "
                        f"<font color=\"red\">{mismatch.mismatch_detail}</font>",
                        s["body"],
                    )
                )
            self.story.append(Spacer(1, 8))

    def _build_issues_section(self):
        """Build comprehensive issues section sorted by severity."""
        s = self.styles
        self.story.append(Paragraph("6. Issues & Findings", s["heading1"]))

        # Collect all issues by severity
        issues_by_severity = {"critical": [], "high": [], "medium": [], "low": []}

        # Circular module dependencies (critical)
        for cycle in self.graph.module_hierarchy.cycles:
            cycle_str = " → ".join(cycle)
            issues_by_severity["critical"].append(
                f"Module cycle: {cycle_str}"
            )

        # Circular includes (high)
        for src, dst in self.graph.include_tree.circular_includes:
            issues_by_severity["high"].append(
                f"Circular include: {src} ↔ {dst}"
            )

        # Unresolved includes (high)
        for unresolved in self.graph.include_tree.unresolved_includes:
            issues_by_severity["high"].append(
                f"Unresolved: {unresolved.source_file}:{unresolved.line} → "
                f"{unresolved.include_name}"
            )

        # High fan-out modules (medium)
        for module in self.graph.module_hierarchy.modules.values():
            if module.fan_out > 15:
                issues_by_severity["medium"].append(
                    f"High fan-out: {module.name} ({module.fan_out} children)"
                )

        # Symbol collisions (medium)
        for symbol_name, _ in self.graph.symbol_table.collisions:
            issues_by_severity["medium"].append(
                f"Symbol collision: {symbol_name}"
            )

        # Parameter type mismatches (medium)
        for mismatch in self.graph.parameter_map.type_mismatches:
            issues_by_severity["medium"].append(
                f"Parameter type mismatch: {mismatch.instance_name}.{mismatch.param_name}"
            )

        # Display issues by severity
        for severity in ["critical", "high", "medium", "low"]:
            issues = issues_by_severity[severity]
            if not issues:
                continue

            color = _severity_color(severity)
            self.story.append(
                Paragraph(
                    f"<font color=\"#{color.hexval()[2:]}\">{severity.upper()}</font> "
                    f"({len(issues)} issue{'s' if len(issues) != 1 else ''})",
                    s["heading2"],
                )
            )
            for issue in issues[:10]:
                self.story.append(Paragraph(f"&bullet;  {issue}", s["body"]))
            if len(issues) > 10:
                self.story.append(
                    Paragraph(f"... and {len(issues) - 10} more", s["body"])
                )
            self.story.append(Spacer(1, 8))

        # Recommendations
        self.story.append(PageBreak())
        self.story.append(Paragraph("Recommendations", s["heading1"]))

        recs = []
        if self.graph.module_hierarchy.cycles:
            recs.append(
                "<b>Resolve module cycles:</b> Use dependency injection, "
                "split modules, or introduce abstraction layers."
            )
        if self.graph.include_tree.circular_includes:
            recs.append(
                "<b>Fix circular includes:</b> Refactor include dependencies, "
                "use include guards, or consolidate headers."
            )
        if self.graph.include_tree.unresolved_includes:
            recs.append(
                "<b>Resolve includes:</b> Verify include paths and file names; "
                "consider using absolute paths or include path variables."
            )
        if any(m.fan_out > 15 for m in self.graph.module_hierarchy.modules.values()):
            recs.append(
                "<b>Reduce fan-out:</b> Hierarchically decompose modules; "
                "group instances into logical sub-blocks."
            )
        if self.graph.symbol_table.total_collisions > 0:
            recs.append(
                "<b>Resolve symbol collisions:</b> Use packages and namespaces; "
                "rename duplicate symbols; use unique prefixes."
            )
        if self.graph.parameter_map.total_mismatches > 0:
            recs.append(
                "<b>Fix parameter mismatches:</b> Verify override types match "
                "parameter definitions; consider type conversion."
            )

        if not recs:
            recs.append(
                "Dependency graph is clean. Continue monitoring as design evolves."
            )

        for rec in recs:
            self.story.append(Paragraph(f"&bullet;  {rec}", s["body"]))

    def _build_methodology_section(self):
        """Build methodology and analysis details."""
        s = self.styles
        self.story.append(PageBreak())
        self.story.append(Paragraph("7. Analysis Methodology", s["heading1"]))

        self.story.append(
            Paragraph(
                "This report was generated using HDL dependency analysis, which extracts "
                "and resolves module hierarchies, include directives, package imports, "
                "parameter overrides, and interface bindings from Verilog/SystemVerilog source.",
                s["body"],
            )
        )
        self.story.append(Spacer(1, 8))

        # Analysis methods table
        header = [
            Paragraph("Component", s["cell_header"]),
            Paragraph("Method", s["cell_header"]),
        ]
        rows = [
            header,
            [
                Paragraph("Module Hierarchy", s["cell_bold"]),
                Paragraph("Regex parsing of module/endmodule; instance instantiation tracking", s["cell"]),
            ],
            [
                Paragraph("Includes", s["cell_bold"]),
                Paragraph("Resolution of `include directives; circular dependency detection", s["cell"]),
            ],
            [
                Paragraph("Packages", s["cell_bold"]),
                Paragraph("SystemVerilog package definition and import resolution", s["cell"]),
            ],
            [
                Paragraph("Parameters", s["cell_bold"]),
                Paragraph("Extraction of parameter overrides in instance declarations", s["cell"]),
            ],
            [
                Paragraph("Interfaces", s["cell_bold"]),
                Paragraph("Interface and modport definition; binding analysis", s["cell"]),
            ],
            [
                Paragraph("Symbols", s["cell_bold"]),
                Paragraph("Cross-file symbol resolution; collision detection", s["cell"]),
            ],
        ]

        tbl = Table(rows, colWidths=[2 * inch, 5 * inch], repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ]
            )
        )
        self.story.append(tbl)
        self.story.append(Spacer(1, 12))

        # Severity definitions
        self.story.append(Paragraph("Issue Severity Definitions", s["heading2"]))
        severity_defs = [
            ("<b>Critical:</b> Module cycles, circular includes", RED),
            ("<b>High:</b> Unresolved includes, unresolved packages", ORANGE),
            ("<b>Medium:</b> High fan-out, symbol collisions, type mismatches", colors.HexColor("#F59E0B")),
            ("<b>Low:</b> Orphan modules, unused symbols", GREEN),
        ]
        for text, _ in severity_defs:
            self.story.append(Paragraph(f"&bullet;  {text}", s["body"]))

    # ── Helper Methods ────────────────────────────────────────────────────

    def _get_module_depth(self, module_name: str) -> Optional[str]:
        """Get the hierarchy depth of a module (simple heuristic)."""
        # Count how many times this module appears in parent-child relationships
        depth = 0
        for inst in self.graph.module_hierarchy.instantiations:
            if inst.child_module == module_name:
                depth += 1
        return str(depth) if depth > 0 else None
