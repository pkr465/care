"""
PDF Complexity Report Generator for CARE.

Produces a professional, multi-page PDF report from HDLComplexityAdapter output.
Uses ReportLab for PDF generation with charts, tables, and per-module breakdowns.

Usage:
    from agents.reports.complexity_report_pdf import ComplexityPDFReport
    report = ComplexityPDFReport(analysis_result, config)
    report.generate("complexity_report.pdf")
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("reports.complexity_pdf")

# ── Colour palette (matches CARE branding) ────────────────────────────
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


# ── Custom styles ─────────────────────────────────────────────────────
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
        "grade_a": ParagraphStyle(
            "GradeA",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#22C55E"),
            alignment=TA_CENTER,
        ),
        "grade_b": ParagraphStyle(
            "GradeB",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#0EA5E9"),
            alignment=TA_CENTER,
        ),
        "grade_c": ParagraphStyle(
            "GradeC",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#F59E0B"),
            alignment=TA_CENTER,
        ),
        "grade_d": ParagraphStyle(
            "GradeD",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#F97316"),
            alignment=TA_CENTER,
        ),
        "grade_f": ParagraphStyle(
            "GradeF",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#EF4444"),
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "CareFooter",
            fontName="Helvetica",
            fontSize=8,
            textColor=MED_GRAY,
            alignment=TA_CENTER,
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
    }
    return styles


# ── Grade helpers ─────────────────────────────────────────────────────
def _grade_style(grade: str, styles: dict) -> str:
    """Return the style name for a grade letter."""
    return styles.get(f"grade_{grade.lower()}", styles["body"])


def _grade_color(grade: str) -> colors.HexColor:
    """Map grade letter to colour."""
    return {
        "A": GREEN,
        "B": BLUE,
        "C": ORANGE,
        "D": colors.HexColor("#F97316"),
        "F": RED,
    }.get(grade.upper(), MED_GRAY)


def _severity_color(severity: str) -> colors.HexColor:
    """Map severity to colour."""
    return {
        "critical": RED,
        "high": ORANGE,
        "medium": colors.HexColor("#F59E0B"),
        "low": GREEN,
    }.get(severity.lower(), MED_GRAY)


def _cc_bar_color(cc: int) -> colors.HexColor:
    """Return colour for CC bar based on threshold."""
    if cc > 20:
        return RED
    if cc > 10:
        return ORANGE
    if cc > 5:
        return BLUE
    return GREEN


# ── Report class ──────────────────────────────────────────────────────
class ComplexityPDFReport:
    """
    Generates a professional PDF complexity report from HDLComplexityAdapter results.

    Args:
        result: The dict returned by HDLComplexityAdapter.analyze()
        config: Optional config dict with project name, etc.
    """

    def __init__(
        self,
        result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.result = result
        self.config = config or {}
        self.styles = _build_styles()
        self.story: List[Any] = []

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, output_path: str) -> str:
        """
        Generate the PDF report and write to *output_path*.

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

        self._build_cover()
        self._build_executive_summary()
        self._build_metrics_dashboard()
        self._build_module_table()
        self._build_flagged_modules()
        self._build_analysis_details()
        self._build_recommendations()

        doc.build(
            self.story,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer,
        )
        logger.info(f"Complexity report written to {output_path}")
        return output_path

    # ── Header / footer ───────────────────────────────────────────────

    def _header_footer(self, canvas, doc):
        """Draw header line and footer on every page."""
        canvas.saveState()
        w, h = letter

        # Header line
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
            1.15 * inch, h - 0.55 * inch, "  |  HDL Complexity Analysis Report"
        )

        # Footer
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MED_GRAY)
        canvas.drawCentredString(
            w / 2, 0.45 * inch, f"Confidential  |  Page {doc.page}"
        )
        date_str = datetime.now().strftime("%B %d, %Y")
        canvas.drawRightString(w - 0.75 * inch, 0.45 * inch, date_str)

        canvas.restoreState()

    # ── Sections ──────────────────────────────────────────────────────

    def _build_cover(self):
        """Build the title / cover section."""
        s = self.styles
        project = self.config.get("project_name", "HDL Design")

        self.story.append(Spacer(1, 0.3 * inch))
        self.story.append(Paragraph("HDL Complexity Analysis Report", s["title"]))
        self.story.append(
            Paragraph(
                f"Project: {project}  |  Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
                s["subtitle"],
            )
        )

        # Analysis mode badge
        metrics = self.result.get("metrics", {})
        mode = metrics.get("analysis_mode", "regex_fallback")
        mode_label = "Verilator AST" if mode == "verilator" else "Regex Fallback"
        self.story.append(
            Paragraph(f"Analysis Mode: <b>{mode_label}</b>", s["body"])
        )
        self.story.append(Spacer(1, 0.2 * inch))

    def _build_executive_summary(self):
        """Build the executive summary with score and grade."""
        s = self.styles
        score = self.result.get("score", 0)
        grade = self.result.get("grade", "F")
        metrics = self.result.get("metrics", {})

        self.story.append(Paragraph("1. Executive Summary", s["heading1"]))

        # Score + grade callout (as a table)
        grade_sty = _grade_style(grade, s)
        grade_clr = _grade_color(grade)

        summary_data = [
            [
                Paragraph(f"{score:.0f}", s["metric_value"]),
                Paragraph(grade, grade_sty),
                Paragraph(
                    f"{metrics.get('modules_analyzed', 0)} modules across "
                    f"{metrics.get('files_analyzed', 0)} files",
                    s["body"],
                ),
            ],
            [
                Paragraph("Score (0-100)", s["metric_label"]),
                Paragraph("Grade", s["metric_label"]),
                Paragraph("Scope", s["metric_label"]),
            ],
        ]

        summary_tbl = Table(summary_data, colWidths=[2 * inch, 1.5 * inch, 3.5 * inch])
        summary_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (1, 1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E5E7EB")),
                ]
            )
        )
        self.story.append(summary_tbl)
        self.story.append(Spacer(1, 12))

        # Issues summary
        issues = self.result.get("issues", [])
        if issues:
            self.story.append(Paragraph("Key Findings:", s["body_bold"]))
            for issue in issues:
                self.story.append(
                    Paragraph(f"&bull;  {issue}", s["body"])
                )
        self.story.append(Spacer(1, 8))

    def _build_metrics_dashboard(self):
        """Build the metrics summary dashboard."""
        s = self.styles
        m = self.result.get("metrics", {})

        self.story.append(Paragraph("2. Metrics Dashboard", s["heading1"]))

        # 4-column metric cards
        card_data = [
            ("Avg CC", f"{m.get('avg_cyclomatic_complexity', 0):.1f}"),
            ("Max CC", f"{m.get('max_cyclomatic_complexity', 0)}"),
            ("Avg Nesting", f"{m.get('avg_nesting_depth', 0):.1f}"),
            ("Avg LOC", f"{m.get('avg_lines_of_code', 0):.0f}"),
        ]

        card_row_vals = [
            [Paragraph(v, s["metric_value"]) for _, v in card_data]
        ]
        card_row_labels = [
            [Paragraph(k, s["metric_label"]) for k, _ in card_data]
        ]

        cards = Table(
            card_row_vals + card_row_labels,
            colWidths=[1.75 * inch] * 4,
        )
        cards.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
                    ("BOX", (0, 0), (0, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BOX", (1, 0), (1, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BOX", (2, 0), (2, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BOX", (3, 0), (3, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
                ]
            )
        )
        self.story.append(cards)
        self.story.append(Spacer(1, 12))

        # Second row: block counts
        block_data = [
            ("Always Blocks", str(m.get("total_always_blocks", 0))),
            ("Always FF", str(m.get("total_always_ff_blocks", 0))),
            ("Always Comb", str(m.get("total_always_comb_blocks", 0))),
            ("High CC Modules", str(m.get("high_cc_count", 0) + m.get("critical_cc_count", 0))),
        ]

        block_vals = [[Paragraph(v, s["metric_value"]) for _, v in block_data]]
        block_labels = [[Paragraph(k, s["metric_label"]) for k, _ in block_data]]

        blocks_tbl = Table(
            block_vals + block_labels,
            colWidths=[1.75 * inch] * 4,
        )
        blocks_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
                    ("BOX", (0, 0), (0, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BOX", (1, 0), (1, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BOX", (2, 0), (2, 1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BOX", (3, 0), (3, 1), 0.5, colors.HexColor("#E5E7EB")),
                ]
            )
        )
        self.story.append(blocks_tbl)
        self.story.append(Spacer(1, 12))

    def _build_module_table(self):
        """Build the full per-module breakdown table."""
        s = self.styles
        details = self.result.get("details", [])
        metrics = self.result.get("metrics", {})

        self.story.append(Paragraph("3. Module Complexity Breakdown", s["heading1"]))
        self.story.append(
            Paragraph(
                f"Analyzed {metrics.get('modules_analyzed', 0)} modules. "
                f"Modules exceeding thresholds are highlighted below.",
                s["body"],
            )
        )

        if not details:
            self.story.append(
                Paragraph(
                    "All modules are within acceptable complexity thresholds.",
                    s["body"],
                )
            )
            return

        # Table header
        header = [
            Paragraph("Module", s["cell_header"]),
            Paragraph("File", s["cell_header"]),
            Paragraph("Line", s["cell_header"]),
            Paragraph("CC", s["cell_header"]),
            Paragraph("Severity", s["cell_header"]),
            Paragraph("Issue", s["cell_header"]),
        ]

        rows = [header]
        for d in details:
            sev = d.get("severity", "low")
            sev_clr = _severity_color(sev)
            rows.append(
                [
                    Paragraph(d.get("module", ""), s["cell_bold"]),
                    Paragraph(d.get("file", ""), s["cell"]),
                    Paragraph(str(d.get("line", "")), s["cell"]),
                    Paragraph(
                        str(
                            self._extract_cc_from_desc(d.get("description", ""))
                        ),
                        s["cell"],
                    ),
                    Paragraph(
                        f'<font color="#{sev_clr.hexval()[2:]}">{sev.upper()}</font>',
                        s["cell"],
                    ),
                    Paragraph(d.get("description", ""), s["cell"]),
                ]
            )

        col_widths = [1.2 * inch, 1.5 * inch, 0.5 * inch, 0.5 * inch, 0.7 * inch, 2.6 * inch]
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)

        # Style
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
        # Alternate row shading
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

        tbl.setStyle(TableStyle(style_cmds))
        self.story.append(tbl)
        self.story.append(Spacer(1, 12))

    def _build_flagged_modules(self):
        """Build a visual bar chart of flagged module CCs."""
        s = self.styles
        details = self.result.get("details", [])

        if not details:
            return

        self.story.append(Paragraph("4. Complexity Distribution", s["heading1"]))
        self.story.append(
            Paragraph(
                "Horizontal bars show cyclomatic complexity for flagged modules. "
                "Thresholds: green (&le;5), blue (6-10), orange (11-20), red (&gt;20).",
                s["body"],
            )
        )

        # Extract module → CC pairs from descriptions
        bars = []
        for d in details:
            cc = self._extract_cc_from_desc(d.get("description", ""))
            name = d.get("module", "unknown")
            if cc > 0:
                bars.append((name, cc))

        if not bars:
            return

        bars.sort(key=lambda x: -x[1])
        bars = bars[:15]  # Top 15

        max_cc = max(cc for _, cc in bars)
        bar_max_w = 4.0 * inch

        rows = []
        for name, cc in bars:
            bar_w = (cc / max(max_cc, 1)) * bar_max_w
            clr = _cc_bar_color(cc)
            # Create a colored cell to represent the bar
            bar_cell = Table(
                [[Paragraph(f"<b>{cc}</b>", s["cell"])]],
                colWidths=[max(bar_w, 0.3 * inch)],
                rowHeights=[16],
            )
            bar_cell.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), clr),
                        ("TEXTCOLOR", (0, 0), (0, 0), WHITE),
                        ("ALIGN", (0, 0), (0, 0), "RIGHT"),
                        ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
                        ("RIGHTPADDING", (0, 0), (0, 0), 4),
                        ("TOPPADDING", (0, 0), (0, 0), 1),
                        ("BOTTOMPADDING", (0, 0), (0, 0), 1),
                    ]
                )
            )
            rows.append(
                [Paragraph(name, s["cell_bold"]), bar_cell]
            )

        chart = Table(rows, colWidths=[2 * inch, 5 * inch])
        chart.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        self.story.append(chart)
        self.story.append(Spacer(1, 12))

    def _build_analysis_details(self):
        """Build the analysis methodology section."""
        s = self.styles
        m = self.result.get("metrics", {})
        mode = m.get("analysis_mode", "regex_fallback")

        self.story.append(Paragraph("5. Analysis Methodology", s["heading1"]))

        if mode == "verilator":
            self.story.append(
                Paragraph(
                    "This report was generated using <b>Verilator AST analysis</b> "
                    "(--xml-only mode). Verilator compiles the RTL into an XML abstract "
                    "syntax tree, from which CARE extracts precise structural metrics:",
                    s["body"],
                )
            )
            methods = [
                ("Cyclomatic Complexity", "Counted from <i>if</i>, <i>case</i>, <i>caseitem</i>, "
                 "<i>while</i>, and <i>for</i> AST nodes within always blocks"),
                ("Nesting Depth", "Measured by walking the AST tree depth through <i>begin</i>, "
                 "<i>if</i>, <i>else</i>, <i>case</i>, and loop structures"),
                ("Port Count", "Extracted from <i>var</i> elements with dir=input/output/inout attributes"),
                ("Signal Count", "Total <i>var</i> elements in the module AST subtree"),
                ("Expression Depth", "Max nesting of operator nodes (add, mul, and, or, etc.) in assignments"),
                ("Operator Count", "Total arithmetic, logical, comparison, and shift operator nodes"),
                ("Instantiation Count", "Number of <i>instance</i> nodes (sub-module instantiations)"),
            ]
        else:
            self.story.append(
                Paragraph(
                    "This report was generated using <b>regex-based pattern matching</b> "
                    "(Verilator not available). Regex analysis provides reliable "
                    "approximations but may not capture all structural nuances:",
                    s["body"],
                )
            )
            methods = [
                ("Cyclomatic Complexity", "Counted from if/else if/case/for/while/ternary keywords"),
                ("Nesting Depth", "Estimated from begin/end brace depth tracking"),
                ("Port Count", "Extracted from module port declaration commas"),
                ("Module LOC", "Lines between module and endmodule keywords"),
            ]

        header = [
            Paragraph("Metric", s["cell_header"]),
            Paragraph("Method", s["cell_header"]),
        ]
        rows = [header]
        for metric, method in methods:
            rows.append(
                [
                    Paragraph(metric, s["cell_bold"]),
                    Paragraph(method, s["cell"]),
                ]
            )

        tbl = Table(rows, colWidths=[2 * inch, 5 * inch], repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
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

        # Thresholds reference
        self.story.append(Paragraph("Complexity Thresholds", s["heading2"]))
        thresh_header = [
            Paragraph("Metric", s["cell_header"]),
            Paragraph("Threshold", s["cell_header"]),
            Paragraph("Action", s["cell_header"]),
        ]
        thresh_rows = [
            thresh_header,
            [
                Paragraph("Cyclomatic Complexity", s["cell"]),
                Paragraph("&gt; 10", s["cell"]),
                Paragraph("Review for refactoring (HIGH)", s["cell"]),
            ],
            [
                Paragraph("Cyclomatic Complexity", s["cell"]),
                Paragraph("&gt; 20", s["cell"]),
                Paragraph("Mandatory refactoring (CRITICAL)", s["cell"]),
            ],
            [
                Paragraph("Nesting Depth", s["cell"]),
                Paragraph("&gt; 4", s["cell"]),
                Paragraph("Extract nested logic into submodules", s["cell"]),
            ],
            [
                Paragraph("Port Count", s["cell"]),
                Paragraph("&gt; 50", s["cell"]),
                Paragraph("Consider interface bundling", s["cell"]),
            ],
            [
                Paragraph("Module LOC", s["cell"]),
                Paragraph("&gt; 300", s["cell"]),
                Paragraph("Split into smaller modules", s["cell"]),
            ],
        ]

        thresh_tbl = Table(thresh_rows, colWidths=[2.2 * inch, 1.3 * inch, 3.5 * inch], repeatRows=1)
        thresh_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                    ("BACKGROUND", (0, 1), (-1, 1), ROW_ALT),
                    ("BACKGROUND", (0, 3), (-1, 3), ROW_ALT),
                    ("BACKGROUND", (0, 5), (-1, 5), ROW_ALT),
                ]
            )
        )
        self.story.append(thresh_tbl)
        self.story.append(Spacer(1, 12))

    def _build_recommendations(self):
        """Build the recommendations section."""
        s = self.styles
        m = self.result.get("metrics", {})

        self.story.append(Paragraph("6. Recommendations", s["heading1"]))

        recs = []
        critical = m.get("critical_cc_count", 0)
        high = m.get("high_cc_count", 0)
        deep = m.get("deep_nesting_count", 0)
        large = m.get("large_module_count", 0)
        many_ports = m.get("many_ports_count", 0)

        if critical > 0:
            recs.append(
                f"<b>Mandatory refactoring:</b> {critical} module(s) have critical "
                f"cyclomatic complexity (&gt;20). Decompose complex always blocks "
                f"into separate combinational and sequential sub-modules."
            )
        if high > 0:
            recs.append(
                f"<b>Recommended refactoring:</b> {high} module(s) have high "
                f"complexity (CC 11-20). Review case statements for simplification "
                f"and extract FSM logic into dedicated state modules."
            )
        if deep > 0:
            recs.append(
                f"<b>Reduce nesting:</b> {deep} module(s) exceed nesting depth 4. "
                f"Flatten nested if/case chains using early-return patterns or "
                f"extract inner logic into tasks/functions."
            )
        if large > 0:
            recs.append(
                f"<b>Reduce module size:</b> {large} module(s) exceed 300 LOC. "
                f"Split into hierarchical sub-modules with clear interfaces."
            )
        if many_ports > 0:
            recs.append(
                f"<b>Bundle interfaces:</b> {many_ports} module(s) have &gt;50 ports. "
                f"Use SystemVerilog interfaces or structs to group related signals."
            )

        if not recs:
            recs.append(
                "All modules are within acceptable complexity thresholds. "
                "Continue monitoring as the design evolves."
            )

        for rec in recs:
            self.story.append(Paragraph(f"&bull;  {rec}", s["body"]))

        self.story.append(Spacer(1, 18))

        # Scoring explanation
        self.story.append(Paragraph("Scoring Methodology", s["heading2"]))
        self.story.append(
            Paragraph(
                "The complexity score starts at 100 and deducts points for each "
                "flagged module: -10 per critical CC module, -5 per high CC module, "
                "-3 per large module, -2 per deep nesting or many-port module. "
                "The final score is clamped to [0, 100] and mapped to a letter grade "
                "(A: 90-100, B: 80-89, C: 70-79, D: 60-69, F: 0-59).",
                s["body"],
            )
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_cc_from_desc(desc: str) -> int:
        """Try to extract CC value from a detail description string."""
        import re

        m = re.search(r"complexity\s*\((\d+)\)", desc)
        if m:
            return int(m.group(1))
        return 0
