#!/usr/bin/env python
"""
main.py

Main entry point and multi-agent workflow for Verilog/SystemVerilog HDL codebase
analysis, design review, flattening to embedding-ready NDJSON, and ingestion into
a PostgreSQL vector DB.

Key stages:
1. PostgreSQL setup (optional, if vector DB enabled)
2. RTL analysis:
   - StaticAnalyzerAgent: unified 7-phase pipeline (optional LLM enrichment)
   - OR CodebaseHDLDesignAgent for strictly Excel/design review reporting
   - Writes canonical designhealth.json
3. Flatten + NDJSON generation:
   - JsonFlattener -> *_flat.json
   - NDJSONProcessor -> *.ndjson (embedding-ready)
4. Vector DB ingestion via VectorDbPipeline

Authors: Pavan R (Original CARE framework), HDL Adaptation Contributors
"""

import os
import re
import argparse
import logging
import sys
import json
import time
import gc
import psutil
import signal
from typing import Dict, List, Any, Optional
from pathlib import Path

from rich.console import Console

console = Console()

# HTML design report generator
try:
    from agents.parsers.healthreport_generator import run_health_report
    HEALTHREPORT_GENERATOR_AVAILABLE = True
except ImportError as e:
    HEALTHREPORT_GENERATOR_AVAILABLE = False
    console.print(f"[yellow]Warning: healthreport_generator not available: {e}[/yellow]")

# Global config support
try:
    from utils.parsers.global_config_parser import GlobalConfig
    GLOBAL_CONFIG_AVAILABLE = True
except ImportError as e:
    GLOBAL_CONFIG_AVAILABLE = False
    console.print(f"[yellow]Warning: GlobalConfig not available: {e}[/yellow]")

# Env and DB setup
from utils.parsers.env_parser import EnvConfig
from db.postgres_db_setup import PostgresDbSetup

# Core analysis agents
from agents.codebase_static_agent import StaticAnalyzerAgent

# Data processing
from utils.data.json_flattener import JsonFlattener
from utils.data.ndjson_processor import NDJSONProcessor

# Vector DB
from utils.data.vector_db_pipeline import VectorDbPipeline

# LLM and utilities
from utils.common.llm_tools import LLMTools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnalysisFramework:
    """Main HDL analysis orchestrator."""

    def __init__(self, args):
        """Initialize the analysis framework with parsed CLI arguments."""
        self.args = args
        self.config = None
        self.llm_tools = None
        self.env_config = None

    def setup(self) -> bool:
        """Setup framework: config, env, DB, LLM."""
        try:
            # Load environment
            self.env_config = EnvConfig()

            # Load global config
            if GLOBAL_CONFIG_AVAILABLE:
                config_file = self.args.config_file
                try:
                    self.config = GlobalConfig(config_file=config_file) if config_file else GlobalConfig()
                except Exception as e:
                    logger.warning(f"Could not load GlobalConfig: {e}")
                    self.config = None

            # Resolve RTL path (CLI > config > default)
            rtl_path = self.args.rtl_path
            if rtl_path == "rtl" and self.config:
                try:
                    config_path = self.config.get_path("paths.code_base_path")
                    if config_path:
                        rtl_path = config_path
                except Exception:
                    pass
            self.rtl_path = Path(rtl_path).resolve()

            # Validate RTL path exists
            if not self.rtl_path.exists():
                console.print(f"[red]Error: RTL path does not exist: {self.rtl_path}[/red]")
                return False

            # Setup output directory
            self.out_dir = Path(self.args.out_dir).resolve()
            os.makedirs(self.out_dir, exist_ok=True)

            # Setup LLM tools if needed
            if self.args.use_llm:
                llm_model = self.args.llm_model
                if not llm_model and self.config:
                    try:
                        llm_model = self.config.get("llm.model")
                    except Exception:
                        llm_model = None

                try:
                    self.llm_tools = LLMTools(model=llm_model) if llm_model else LLMTools()
                except Exception as e:
                    logger.warning(f"Could not initialize LLM tools: {e}")
                    if not self.args.llm_exclusive:
                        logger.info("Continuing without LLM support")
                    else:
                        console.print(f"[red]Error: LLM required but initialization failed: {e}[/red]")
                        return False

            # Setup PostgreSQL if vector DB enabled
            if self.args.enable_vector_db:
                try:
                    db_setup = PostgresDbSetup(config=self.config)
                    if not db_setup.setup():
                        logger.warning("PostgreSQL setup failed; vector DB will be skipped")
                        self.args.enable_vector_db = False
                except Exception as e:
                    logger.warning(f"PostgreSQL setup error: {e}")
                    self.args.enable_vector_db = False

            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=self.args.debug)
            return False

    def run(self) -> bool:
        """Execute the HDL analysis pipeline."""
        try:
            if not self.setup():
                return False

            console.print("\n" + "=" * 70)
            console.print("[bold cyan]CARE — HDL Analysis & Repair Engine[/bold cyan]")
            console.print("=" * 70 + "\n")

            console.print(f"[cyan]RTL Path:[/cyan]         {self.rtl_path}")
            console.print(f"[cyan]Output Directory:[/cyan] {self.out_dir}")
            console.print(f"[cyan]LLM Enabled:[/cyan]      {self.args.use_llm}")
            console.print(f"[cyan]Vector DB:[/cyan]        {self.args.enable_vector_db}")
            console.print(f"[cyan]Deep Analysis:[/cyan]    {self.args.enable_deep_analysis}")

            # Phase 1: HDL Analysis
            console.print("\n[bold green]Phase 1: RTL Analysis[/bold green]")
            design_health = self._run_analysis()
            if not design_health:
                console.print("[red]Analysis failed[/red]")
                return False

            # Phase 2: Data flattening
            console.print("\n[bold green]Phase 2: Data Flattening[/bold green]")
            flat_data = self._flatten_data(design_health)
            if not flat_data:
                console.print("[red]Flattening failed[/red]")
                return False

            # Phase 3: NDJSON processing
            console.print("\n[bold green]Phase 3: NDJSON Processing[/bold green]")
            ndjson_file = self._process_ndjson(flat_data)
            if not ndjson_file:
                console.print("[red]NDJSON processing failed[/red]")
                return False

            # Phase 4: Vector DB ingestion
            if self.args.enable_vector_db:
                console.print("\n[bold green]Phase 4: Vector DB Ingestion[/bold green]")
                if not self._ingest_vector_db(ndjson_file):
                    console.print("[yellow]Vector DB ingestion skipped[/yellow]")

            # Phase 5: Report generation
            console.print("\n[bold green]Phase 5: Report Generation[/bold green]")
            self._generate_reports(design_health)

            console.print("\n" + "=" * 70)
            console.print("[bold green]Analysis Complete[/bold green]")
            console.print("=" * 70)
            console.print(f"\nOutputs in: {self.out_dir}")
            console.print(f"  - designhealth.json        : Canonical design metrics")
            console.print(f"  - design_review.xlsx       : Design review spreadsheet")
            console.print(f"  - diagrams/                : Hierarchy visualizations")

            return True

        except KeyboardInterrupt:
            console.print("\n[yellow]Analysis interrupted by user[/yellow]")
            return False
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=self.args.debug)
            return False

    def _run_analysis(self) -> Optional[Dict[str, Any]]:
        """Execute HDL analysis using StaticAnalyzerAgent."""
        try:
            analyzer = StaticAnalyzerAgent(
                codebase_path=str(self.rtl_path),
                output_dir=str(self.out_dir),
                config=self.config,
                llm_tools=self.llm_tools if self.args.use_llm else None,
                enable_adapters=self.args.enable_deep_analysis,
                verbose=self.args.verbose,
                debug=self.args.debug,
            )

            # Run analysis
            design_health = analyzer.run_analysis()

            if design_health:
                # Save canonical report
                report_path = self.out_dir / "designhealth.json"
                with open(report_path, 'w') as f:
                    json.dump(design_health, f, indent=2)
                console.print(f"[green]✓[/green] Analysis complete. Report: {report_path}")
                return design_health
            else:
                console.print("[red]✗[/red] Analysis returned no results")
                return None

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=self.args.debug)
            return None

    def _flatten_data(self, design_health: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Flatten hierarchical design health report."""
        try:
            flattener = JsonFlattener()
            flat_data = flattener.flatten(design_health)

            # Save flat JSON
            flat_path = self.out_dir / "designhealth_flat.json"
            with open(flat_path, 'w') as f:
                json.dump(flat_data, f, indent=2)

            console.print(f"[green]✓[/green] Flattened to: {flat_path}")
            return flat_data

        except Exception as e:
            logger.error(f"Flattening error: {e}", exc_info=self.args.debug)
            return None

    def _process_ndjson(self, flat_data: Dict[str, Any]) -> Optional[Path]:
        """Convert flat data to NDJSON format."""
        try:
            processor = NDJSONProcessor(
                chunk_size=self.args.vector_chunk_size,
                overlap_size=self.args.vector_overlap_size,
            )
            ndjson_records = processor.process(flat_data)

            # Save NDJSON
            ndjson_path = self.out_dir / "designhealth.ndjson"
            with open(ndjson_path, 'w') as f:
                for record in ndjson_records:
                    f.write(json.dumps(record) + '\n')

            console.print(f"[green]✓[/green] NDJSON: {ndjson_path} ({len(ndjson_records)} records)")
            return ndjson_path

        except Exception as e:
            logger.error(f"NDJSON processing error: {e}", exc_info=self.args.debug)
            return None

    def _ingest_vector_db(self, ndjson_file: Path) -> bool:
        """Ingest NDJSON records into PostgreSQL vector DB."""
        try:
            pipeline = VectorDbPipeline(config=self.config)
            success, count = pipeline.ingest_ndjson(str(ndjson_file))

            if success:
                console.print(f"[green]✓[/green] Ingested {count} records into vector DB")
                return True
            else:
                console.print(f"[yellow]⚠[/yellow] Vector DB ingestion incomplete")
                return False

        except Exception as e:
            logger.error(f"Vector DB ingestion error: {e}", exc_info=self.args.debug)
            return False

    def _generate_reports(self, design_health: Dict[str, Any]) -> None:
        """Generate HTML and other reports."""
        try:
            if HEALTHREPORT_GENERATOR_AVAILABLE:
                html_path = self.out_dir / "design_health.html"
                run_health_report(design_health, output_path=str(html_path))
                console.print(f"[green]✓[/green] HTML report: {html_path}")
        except Exception as e:
            logger.warning(f"Report generation skipped: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CARE: Codebase Analysis & Repair Engine for HDL (Verilog/SystemVerilog)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ─────────────────────────────────────────────────────────────────────────
    # RTL/Source Path Options
    # ─────────────────────────────────────────────────────────────────────────
    rtl_group = parser.add_argument_group("RTL/Source Path")
    rtl_group.add_argument(
        "--rtl-path",
        default="./rtl",
        help="Root directory of the RTL source code"
    )
    rtl_group.add_argument(
        "--codebase-path",
        default="./rtl",
        help="(Alias for --rtl-path) Root directory of the RTL source code"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Output & Configuration
    # ─────────────────────────────────────────────────────────────────────────
    config_group = parser.add_argument_group("Output & Configuration")
    config_group.add_argument(
        "--out-dir",
        default="./out",
        help="Directory for output files"
    )
    config_group.add_argument(
        "--config-file",
        default=None,
        help="Path to global_config.yaml (overrides default)"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HDL Analysis Options
    # ─────────────────────────────────────────────────────────────────────────
    analysis_group = parser.add_argument_group("HDL Analysis")
    analysis_group.add_argument(
        "--use-verible",
        action="store_true",
        help="Enable Verible parser integration for syntax checking"
    )
    analysis_group.add_argument(
        "--enable-deep-analysis",
        action="store_true",
        help="Enable deep HDL analysis adapters (Verible, Verilator, Yosys)"
    )
    analysis_group.add_argument(
        "--target-technology",
        choices=["fpga", "asic"],
        default="fpga",
        help="Target technology (fpga or asic) for synthesis context"
    )
    analysis_group.add_argument(
        "--clock-period",
        type=float,
        default=10.0,
        metavar="NS",
        help="Target clock period in nanoseconds (for timing analysis hints)"
    )
    analysis_group.add_argument(
        "--reset-strategy",
        choices=["async", "sync"],
        default="async",
        help="Reset architecture strategy (async or sync)"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Configuration
    # ─────────────────────────────────────────────────────────────────────────
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM-powered design review phase"
    )
    llm_group.add_argument(
        "--llm-exclusive",
        action="store_true",
        help="Run LLM analysis only (skip static analyzer)"
    )
    llm_group.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help="LLM model in 'provider::model' format "
             "(e.g., 'anthropic::claude-sonnet-4-20250514')"
    )
    llm_group.add_argument(
        "--llm-api-key",
        default=None,
        help="LLM API Key (overrides env vars)"
    )
    llm_group.add_argument(
        "--llm-max-tokens",
        type=int,
        default=16384,
        help="Token limit for LLM requests"
    )
    llm_group.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for LLM"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Vector DB Options
    # ─────────────────────────────────────────────────────────────────────────
    vectordb_group = parser.add_argument_group("Vector DB (PostgreSQL)")
    vectordb_group.add_argument(
        "--enable-vector-db",
        action="store_true",
        help="Ingest analysis results into PostgreSQL vector database"
    )
    vectordb_group.add_argument(
        "--vector-chunk-size",
        type=int,
        default=512,
        help="Characters per embedding chunk"
    )
    vectordb_group.add_argument(
        "--vector-overlap-size",
        type=int,
        default=128,
        help="Character overlap between chunks"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HITL & Feedback Options
    # ─────────────────────────────────────────────────────────────────────────
    hitl_group = parser.add_argument_group("HITL (Human-in-the-Loop)")
    hitl_group.add_argument(
        "--enable-hitl",
        action="store_true",
        help="Enable human-in-the-loop feedback store"
    )
    hitl_group.add_argument(
        "--hitl-feedback-excel",
        default=None,
        help="Excel file with human design reviews"
    )
    hitl_group.add_argument(
        "--hitl-constraints-dir",
        default=None,
        help="Directory with design rule markdown files"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Output & Reporting
    # ─────────────────────────────────────────────────────────────────────────
    report_group = parser.add_argument_group("Output & Reporting")
    report_group.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit analysis to first N files"
    )
    report_group.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for LLM processing"
    )
    report_group.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        metavar="MB",
        help="Memory limit for analysis"
    )
    report_group.add_argument(
        "--generate-visualizations",
        action="store_true",
        help="Create module hierarchy diagrams"
    )
    report_group.add_argument(
        "--generate-pdfs",
        action="store_true",
        help="Generate PDF design documentation"
    )
    report_group.add_argument(
        "--generate-report",
        action="store_true",
        help="Create HTML design health report"
    )
    report_group.add_argument(
        "--force-reanalysis",
        action="store_true",
        help="Re-analyze all files (ignore cache)"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # File Scanning Options
    # ─────────────────────────────────────────────────────────────────────────
    scan_group = parser.add_argument_group("File Scanning")
    scan_group.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=[],
        metavar="DIR",
        help="Additional directories to exclude"
    )
    scan_group.add_argument(
        "--exclude-globs",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="Additional glob patterns to exclude"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Debugging & Verbosity
    # ─────────────────────────────────────────────────────────────────────────
    debug_group = parser.add_argument_group("Debugging & Verbosity")
    debug_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    debug_group.add_argument(
        "-D", "--debug",
        action="store_true",
        help="Enable debug mode with full tracebacks"
    )

    args = parser.parse_args()

    # Validate RTL path argument resolution
    if args.rtl_path == "./rtl" and args.codebase_path != "./rtl":
        args.rtl_path = args.codebase_path

    # Setup logging level
    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run framework
    framework = AnalysisFramework(args)
    success = framework.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
