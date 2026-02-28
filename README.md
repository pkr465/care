# CARE — Codebase Analysis & Repair Engine for HDL

Multi-agent static analysis and AI-assisted design review framework for **Verilog/SystemVerilog** hardware design codebases. CARE provides a complete pipeline for analyzing RTL (Register Transfer Level) designs, generating rich health metrics, structural hierarchy analysis, and design quality reports suitable for HDL design teams.

## Key Capabilities

1. **Static HDL Analysis**: Uses a unified `StaticAnalyzerAgent` (7-phase pipeline) with specialized Verilog/SystemVerilog analyzers for design quality scoring.
2. **Deep HDL Analysis Adapters**: Optional `--enable-deep-analysis` mode powered by Verible, Verilator, and Yosys for AST-accurate metrics and design rule checking.
3. **LLM-Powered Design Review**: `CodebaseHDLDesignAgent` performs per-module semantic analysis and produces `design_review.xlsx`.
4. **Design Health Reporting**: Produces a canonical `designhealth.json` with hierarchy, complexity, and design rule metrics; optional HTML rendering.
5. **Data Flattening**: Converts reports to JSON and NDJSON formats for embedding.
6. **Vector DB Ingestion**: Ingests design data into a **PostgreSQL** vector database with pgvector for semantic search.
7. **Agentic Design Repair**: Human-in-the-loop `CodebaseDesignRepairAgent` applies LLM-suggested design fixes guided by reviewer feedback.
8. **Multi-Provider LLM**: Supports Anthropic Claude, QGenie, Google Vertex AI, and Azure OpenAI via `provider::model` format.
9. **Interactive Dashboard**: Professional silicon design website (`index.html`) plus a dark-themed Streamlit UI with real-time pipeline monitoring, chat, and telemetry.
10. **Telemetry & Analytics**: Silent PostgreSQL-backed telemetry tracks design issues found/fixed, LLM usage, run durations, and fix success rates with a built-in dashboard.
11. **HITL Feedback Store**: PostgreSQL-backed persistent store for human review decisions and design rules, enabling agents to learn from accumulated design review history.
12. **Batch Design Patch Agent**: Applies multi-file design patches (with `===` file headers) to a local RTL codebase, producing patched copies in `out/patched_designs/` with folder structure preserved.

---

## Quick Start

### One-Command Installation

```bash
git clone <repo>
cd CARE
chmod +x install.sh
./install.sh
```

The installer automatically handles everything across **macOS**, **Linux**, and **Windows (WSL)**:

- Detects your OS and package manager (Homebrew, apt, dnf, yum, pacman)
- Installs Python 3.9+ if needed
- Creates a `.venv` virtual environment
- Installs all Python dependencies from `requirements.txt`
- Installs HDL tools: Verible, Verilator, Yosys, Icarus Verilog
- Installs optional tools: Pandoc, Mermaid CLI
- Sets up `.env` from `env.example`
- Validates the full installation

You can skip parts with environment variables:

```bash
CARE_SKIP_HDL=1 ./install.sh        # Skip HDL tools
CARE_SKIP_DB=1 ./install.sh         # Skip database setup info
CARE_SKIP_OPTIONAL=1 ./install.sh   # Skip Pandoc, mmdc
CARE_PYTHON=python3.11 ./install.sh # Override Python binary
```

### Launch the Dashboard

```bash
./launch.sh                # Start Streamlit dashboard on port 8502
./launch.sh --website      # Also open the silicon design website
./launch.sh --port 8503    # Custom port
```

The launch script activates the virtual environment, validates dependencies, and starts the Streamlit dashboard. Access it at `http://localhost:8502`.

### Open the Silicon Design Website

```bash
open index.html       # macOS
xdg-open index.html   # Linux
```

The website provides a professional overview of CARE's capabilities. The **Start Analysis** button launches the Streamlit dashboard directly.

### Run CLI Analysis

```bash
source .venv/bin/activate
export LLM_API_KEY="sk-..."

# Full pipeline
python main.py --rtl-path ./rtl --out-dir ./out --use-llm

# LLM-exclusive design review
python main.py --rtl-path ./rtl --llm-exclusive --use-llm
```

### Set Up PostgreSQL (Optional)

The database enables vector search, telemetry, and HITL feedback. Core analysis works without it.

```bash
sudo ./bootstrap_db.sh                          # Local setup
DB_HOST=db.example.com ./bootstrap_db.sh        # Remote setup
```

---

## Architecture & Workflow

### Standard Workflow (LangGraph)

The workflow is orchestrated using **LangGraph** and consists of four main agents:

1. **PostgreSQL Setup** (`postgres_db_setup_agent`): Sets up the schema and tables for vector storage.
2. **RTL Analysis** (`rtl_analysis_agent`): Runs `StaticAnalyzerAgent` (7-phase pipeline with optional LLM enrichment and deep adapters). Generates `designhealth.json`.
3. **Flatten & NDJSON** (`flatten_and_ndjson_agent`): Flattens the report (`JsonFlattener`) and converts it to NDJSON (`NDJSONProcessor`) for embedding processing.
4. **Vector DB Ingestion** (`vector_db_ingestion_agent`): Ingests the processed records into PostgreSQL via `VectorDbPipeline`.

```text
PostgreSQL Setup
    ↓
RTL Analysis (StaticAnalyzerAgent — 7 analyzers + optional deep adapters)
    ↓
Flatten & NDJSON
    ↓
Vector DB Ingestion
```

### Exclusive LLM Mode (`--llm-exclusive`)

Bypasses the LangGraph workflow entirely. Runs `CodebaseHDLDesignAgent` for per-module semantic analysis producing `design_review.xlsx`. When combined with `--enable-deep-analysis`, deep adapter results (hierarchy, design rules, complexity, timing) are merged as `static_*` tabs in the same Excel file.

```text
[Optional] Deep HDL Analysis Adapters (Verible, Verilator, Yosys)
    ↓
CodebaseHDLDesignAgent (per-module LLM analysis)
    ↓
design_review.xlsx (LLM tabs + static_ adapter tabs)
```

### Deep HDL Analysis Adapters (`--enable-deep-analysis`)

When enabled, the following adapters run using real HDL tooling instead of regex heuristics:

| Adapter                      | Backend    | Capabilities                                                 |
| :--------------------------- | :--------- | :----------------------------------------------------------- |
| `HDLComplexityAdapter`       | Verilator  | Cyclomatic complexity, nesting depth, expression depth via AST |
| `CallGraphAdapter`           | Verilator  | Module instantiation graph, fan-in/fan-out analysis          |
| `DeadCodeAdapter`            | Verilator  | Dead signal detection via elaboration                        |
| `FunctionMetricsAdapter`     | Verilator  | Port widths, parameter ranges, type checking                 |
| `SecurityAdapter`            | Regex      | HDL security vulnerability pattern analysis                  |
| `DependencyGraphAdapter`     | Regex/Verible | Module hierarchy, include tree, package imports, symbol table |
| `ExcelReportAdapter`         | ExcelWriter| Generates `static_*` prefixed tabs in Excel output           |

All adapters inherit from `BaseStaticAdapter` and degrade gracefully when their underlying tool is unavailable.

### Project Layout

```text
.
├── install.sh                          # One-command cross-platform installer
├── launch.sh                           # Dashboard launcher (venv + Streamlit)
├── index.html                   # Silicon design website (dark theme)
├── main.py                             # Entry point & LangGraph workflow
├── fixer_workflow.py                   # Human-in-the-loop design repair workflow
├── global_config.yaml                  # Hierarchical YAML configuration
├── requirements.txt                    # Python dependencies
├── bootstrap_db.sh                     # PostgreSQL + pgvector setup script
├── env.example                         # Environment variable template
├── agents/
│   ├── codebase_static_agent.py        # Unified 7-phase HDL analyzer
│   ├── codebase_hdl_design_agent.py    # LLM-exclusive per-module design reviewer
│   ├── codebase_batch_design_patch_agent.py    # Batch multi-file design patch application
│   ├── codebase_design_repair_agent.py  # Agentic design repair agent (source-aware, audit trail)
│   ├── codebase_design_patch_agent.py  # Design patch analysis agent (diff-based issue detection)
│   ├── codebase_analysis_chat_agent.py # Interactive chat design analysis agent
│   ├── adapters/                       # Deep HDL analysis adapters
│   │   ├── base_adapter.py             #   ABC base class
│   │   ├── ast_complexity_adapter.py   #   Verilator AST complexity (CC, nesting, expression depth)
│   │   ├── call_graph_adapter.py       #   Verilator module instantiation graph
│   │   ├── dead_code_adapter.py        #   Verilator dead signal detection
│   │   ├── function_metrics_adapter.py #   Verilator port/parameter metrics
│   │   ├── security_adapter.py         #   HDL security pattern analysis
│   │   ├── dependency_graph_adapter.py #   HDL dependency graph scoring
│   │   └── excel_report_adapter.py     #   static_ Excel tab generator
│   ├── services/                       # HDL dependency analysis services
│   │   ├── module_hierarchy_builder.py #   Module instantiation tree (networkx)
│   │   ├── include_dependency_graph.py #   `include file resolution
│   │   ├── package_import_resolver.py  #   Package import chain resolution
│   │   ├── parameter_propagation_tracker.py # Parameter override tracking
│   │   ├── interface_binding_analyzer.py #  Interface/modport binding graph
│   │   ├── generate_block_expander.py  #   Generate block conditional tracking
│   │   └── symbol_table_builder.py     #   Cross-file symbol resolution
│   ├── context/                        # Context layers for LLM analysis
│   │   ├── __init__.py
│   │   └── header_context_builder.py   #   Include resolution, macro parsing
│   ├── analyzers/                      # HDL health analyzers
│   │   ├── base_runtime_analyzer.py    #   Base class for runtime analyzers
│   │   ├── dependency_analyzer.py      #   HDL dependency analysis orchestrator
│   │   ├── quality_analyzer.py         #   Design quality scoring
│   │   ├── complexity_analyzer.py      #   Combinational & sequential logic complexity
│   │   ├── cdc_analyzer.py             #   Clock domain crossing analysis
│   │   ├── documentation_analyzer.py   #   Comment coverage and design docs
│   │   ├── maintainability_analyzer.py #   Code style and readability
│   │   ├── memory_corruption_analyzer.py #  Memory/FIFO corruption detection
│   │   ├── null_pointer_analyzer.py    #   Null/X signal propagation
│   │   ├── potential_deadlock_analyzer.py # FSM deadlock detection
│   │   ├── security_analyzer.py        #   Security vulnerability patterns
│   │   ├── signal_integrity_analyzer.py #  Signal integrity analysis
│   │   ├── synthesis_safety_analyzer.py #  Synthesis safety checks
│   │   ├── test_coverage_analyzer.py   #   Test/verification coverage
│   │   ├── uninitialized_signal_analyzer.py # Uninitialized signal detection
│   │   └── verification_coverage_analyzer.py # Formal verification coverage
│   ├── reports/                        # Report generators
│   │   ├── complexity_report_pdf.py    #   ReportLab PDF complexity report
│   │   └── dependency_report_pdf.py    #   ReportLab PDF dependency report
│   └── parsers/
│       ├── excel_to_agent_parser.py    #   Parse Excel design review → directives.jsonl
│       ├── healthreport_generator.py   #   HTML health report generation
│       └── healthreport_parser.py      #   Health report parsing
├── utils/
│   ├── common/
│   │   ├── llm_tools.py                #   Multi-provider LLM wrapper (Anthropic/QGenie/VertexAI/Azure)
│   │   └── llm_tools_*.py              #   Provider-specific implementations
│   ├── parsers/
│   │   ├── global_config_parser.py     #   YAML config + env var resolution
│   │   └── env_parser.py               #   .env file loader
│   └── data/
│       ├── json_flattener.py           #   Report → JSON flattening
│       ├── ndjson_processor.py         #   JSON → NDJSON conversion
│       ├── vector_db_pipeline.py       #   Ingest into PostgreSQL/pgvector
│       ├── hitl_feedback_store.py      #   Human-in-the-loop decision log
│       └── metrics_calculator.py       #   Aggregate adapter metrics
├── db/
│   └── postgres_db_setup.py            #   PostgreSQL schema creation
├── ui/
│   ├── app.py                          #   Streamlit dashboard (dark silicon theme)
│   ├── launch.py                       #   Python-based Streamlit launcher
│   ├── streamlit_tools.py              #   Shared UI helpers, CSS, widgets
│   ├── background_workers.py           #   Async analysis/fixer workers
│   ├── feedback_helpers.py             #   Excel export & feedback utilities
│   └── qa_inspector.py                 #   QA traceability inspector
├── hitl/                               # Human-in-the-loop feedback pipeline
├── prompts/                            # LLM system & user prompts
└── out/                                # Output directory
    ├── designhealth.json               #   Canonical design health report
    ├── design_review.xlsx              #   Human review spreadsheet
    ├── patched_designs/                #   Design patch results
    ├── diagrams/                       #   Module hierarchy diagrams
    ├── pdfs/                           #   Design documentation PDFs
    └── parseddata/                     #   Flattened JSON & NDJSON
```

---

## Configuration

### global_config.yaml

Hierarchical YAML configuration file supports environment variable overrides via `${ENV_VAR}` syntax.

Key sections:

- **paths**: RTL source directory, output directories, prompt templates
- **llm**: LLM provider (anthropic, qgenie, vertexai, azure), model selection, API keys
- **database**: PostgreSQL connection for vector DB and telemetry
- **scanning**: RTL-specific exclusions (.Xil, sim_results, synthesis, etc.)
- **hierarchy_builder**: Verible and Verilator configuration
- **context**: Include file resolution for `include path context injection
- **synthesis**: Target technology, clock period, reset strategy for timing hints
- **eda_tools**: Paths to Verible, Verilator, Yosys, Icarus Verilog
- **hitl**: Human-in-the-loop feedback store configuration
- **telemetry**: Silent usage tracking for RTL analysis metrics
- **email**: Report delivery via SMTP
- **excel**: Design review spreadsheet styling

### Environment Variables (.env)

The `.env` file holds only LLM API keys. Everything else (database, SMTP, paths) is in `global_config.yaml`.

```bash
cp env.example .env
# Then edit .env with your API keys
```

```bash
LLM_API_KEY=sk-...                     # Required for LLM analysis (any provider)
QGENIE_API_KEY=...                     # Optional, if using QGenie models
```

---

## Command-Line Options

### Core Options

```
--rtl-path PATH                Root directory of RTL source code (default: ./rtl)
--codebase-path PATH           Alias for --rtl-path
--out-dir DIR                  Output directory (default: ./out)
--config-file FILE             Custom global_config.yaml path
```

### HDL Analysis Options

```
--use-verible                  Enable Verible parser integration
--enable-deep-analysis         Enable Verible/Verilator deep analysis adapters
--target-technology TYPE       fpga | asic (for synthesis context)
--clock-period FLOAT           Target clock period in ns (for timing hints)
--reset-strategy STRATEGY      async | sync (reset architecture hint)
```

### LLM Options

```
--use-llm                      Enable LLM analysis phase
--llm-exclusive                Run LLM analysis only (skip static analyzer)
--llm-model MODEL              Model in 'provider::name' format
--llm-api-key KEY              API Key override
--llm-max-tokens TOKENS        Token limit (default: 16384)
--llm-temperature TEMP         Sampling temperature (default: 0.1)
```

### Vector DB Options

```
--enable-vector-db             Ingest results into PostgreSQL
--vector-chunk-size INT        Characters per chunk (default: 512)
--vector-overlap-size INT      Overlap between chunks (default: 128)
```

### HITL & Feedback Options

```
--enable-hitl                  Enable human-in-the-loop feedback store
--hitl-feedback-excel FILE     Excel file with human reviews
--hitl-constraints-dir DIR     Directory with design rule markdown files
```

### Output & Reporting

```
--generate-visualizations      Create module hierarchy diagrams
--generate-pdfs                Generate PDF design documentation
--generate-report              Create HTML design health report
--max-files INT                Limit analysis to N files
--batch-size INT               Batch size for LLM processing (default: 5)
--memory-limit MB              Memory limit for large analyses
```

### Miscellaneous

```
--force-reanalysis             Re-analyze all files (ignore cache)
--exclude-dirs DIR [DIR ...]   Additional directories to skip
--exclude-globs GLOB [GLOB ...] Additional glob patterns to skip
-v, --verbose                  Enable detailed logging
-D, --debug                    Enable debug mode
```

---

## Data Flow Pipeline

### Phase 1: CLI Argument Parsing
Processes command-line flags and resolves configuration from CLI arguments (highest priority), `global_config.yaml`, `.env` environment variables, and built-in defaults.

### Phase 2: Environment & Config Loading
Loads `global_config.yaml` with environment variable substitution (e.g., `${LLM_API_KEY}`). Validates RTL path, output directory, and tool availability.

### Phase 3: PostgreSQL Database Setup (optional)
If `--enable-vector-db`, creates schema, tables, and pgvector extension in PostgreSQL for vector embeddings.

### Phase 4: HDL Codebase Analysis
Executes `StaticAnalyzerAgent` (7-phase pipeline) with 7 regex-based analyzers. Optionally runs deep analysis adapters (Verible, Verilator) and `CodebaseHDLDesignAgent` for LLM design review. Produces canonical `designhealth.json`.

### Phase 5: JSON Flattening
Converts `designhealth.json` to flat JSON records using `JsonFlattener`. One record per design element (module, port, parameter, etc.).

### Phase 6: NDJSON Processing
Converts flat JSON to NDJSON (newline-delimited JSON) using `NDJSONProcessor`. Suitable for streaming ingestion into vector DB.

### Phase 7: Vector DB Ingestion
If `--enable-vector-db`, ingests NDJSON records into PostgreSQL with pgvector embeddings via `VectorDbPipeline`. Enables semantic search of design elements.

### Phase 8: Interactive Dashboard
Launch the Streamlit UI via `./launch.sh` for interactive design Q&A, real-time pipeline monitoring, telemetry dashboard, and feedback collection.

---

## Examples

### Analyze RTL with LLM Review

```bash
python main.py \
  --rtl-path ./designs/my_soc \
  --out-dir ./results \
  --use-llm \
  --llm-model "anthropic::claude-sonnet-4-20250514" \
  --generate-visualizations \
  -v
```

Output: `results/design_review.xlsx`, `results/designhealth.json`, module hierarchy diagrams.

### Deep Analysis with All Adapters

```bash
python main.py \
  --rtl-path ./rtl \
  --enable-deep-analysis \
  --llm-exclusive \
  --target-technology fpga \
  --target-device xcvu9p \
  --clock-period 5.0
```

Output: `out/design_review.xlsx` with deep metrics (hierarchy, design rules, complexity).

### Vector DB Ingestion for Design Search

```bash
python main.py \
  --rtl-path ./rtl \
  --enable-vector-db \
  --use-llm \
  --vector-chunk-size 1024 \
  --vector-overlap-size 256
```

Then launch the dashboard:
```bash
./launch.sh
```

### Design Repair Workflow

```bash
# Step 1: Generate design review
python main.py --rtl-path ./rtl --llm-exclusive -v

# Step 2: Open out/design_review.xlsx, review, and mark issues

# Step 3: Apply repairs
python fixer_workflow.py \
  --excel-file ./out/design_review.xlsx \
  --rtl-path ./rtl \
  --llm-model "anthropic::claude-sonnet-4-20250514"
```

---

## Troubleshooting

### Installation Issues

Run the installer with verbose output to diagnose problems:

```bash
bash -x ./install.sh 2>&1 | tee install.log
```

On Windows, make sure you're running inside WSL:

```powershell
wsl --install          # Install WSL if needed
wsl                    # Enter WSL
cd /path/to/CARE
./install.sh
```

### Verible/Verilator Not Found

Ensure HDL tools are installed and in PATH:

```bash
# Ubuntu/Debian
sudo apt-get install verible verilator

# macOS
brew install verible verilator

# Verify
which verible-verilog-syntax
which verilator
```

### PostgreSQL Connection Error

Check database credentials in `global_config.yaml` and `.env`:

```bash
psql -h localhost -U codebase_analytics_user -d codebase_analytics_db
```

### Out of Memory

Use `--memory-limit` or `--batch-size` to reduce memory footprint:

```bash
python main.py --rtl-path ./rtl --batch-size 2 --memory-limit 4096
```

### Dashboard Won't Start

```bash
# Check Streamlit is installed
source .venv/bin/activate
python -c "import streamlit; print(streamlit.__version__)"

# Check port availability
lsof -i :8502

# Try a different port
./launch.sh --port 8503
```

---

## Contributing

Contributions welcome! Please follow:

1. Create feature branch: `git checkout -b feature/my-feature`
2. Commit with clear messages: `git commit -m "Add HDL feature"`
3. Push and open PR: `git push origin feature/my-feature`

---

## License

See LICENSE file for terms.

---

## Authors

- Pavan R (CARE framework — Verilog/SystemVerilog HDL)

---

## Contact

For issues, feature requests, or questions, please open a GitHub issue or contact the maintainers.
