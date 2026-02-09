# QGenie — C/C++ Codebase Health Analysis & Refactor Engine

`main.py` is the main entry point for a **multi-stage C/C++ codebase health analysis pipeline**. It is designed to analyze C/C++ codebases and produce rich health scores, structural metadata, and embeddings suitable for RAG (Retrieval-Augmented Generation) applications.

## Key Capabilities

1. **Static Analysis**: Uses a unified `StaticAnalyzerAgent` (7-phase pipeline) with 9 regex-based analyzers for fast health scoring.
2. **Deep Static Analysis Adapters**: Optional `--enable-adapters` mode powered by CCLS/libclang, Lizard, and Flawfinder for AST-accurate metrics.
3. **LLM-Powered Code Review**: `CodebaseLLMAgent` performs per-file semantic analysis and produces `detailed_code_review.xlsx`.
4. **Health Reporting**: Produces a canonical `healthreport.json` with metrics and summaries; optional HTML rendering.
5. **Data Flattening**: Converts reports to JSON and NDJSON formats for embedding.
6. **Vector DB Ingestion**: Ingests data into a **PostgreSQL** vector database with pgvector.
7. **Agentic Code Repair**: Human-in-the-loop `CodebaseFixerAgent` applies LLM-suggested fixes guided by reviewer feedback.
8. **Multi-Provider LLM**: Supports Anthropic Claude, QGenie, Google Vertex AI, and Azure OpenAI via `provider::model` format.
9. **Visualization**: Generates an HTML health report and provides a Streamlit UI dashboard.

---

## Architecture & Workflow

### Standard Workflow (LangGraph)

The workflow is orchestrated using **LangGraph** and consists of four main agents:

1. **PostgreSQL Setup** (`postgres_db_setup_agent`): Sets up the schema and tables for vector storage.
2. **Codebase Analysis** (`codebase_analysis_agent`): Runs `StaticAnalyzerAgent` (7-phase pipeline with optional LLM enrichment and deep adapters). Generates `healthreport.json`.
3. **Flatten & NDJSON** (`flatten_and_ndjson_agent`): Flattens the report (`JsonFlattener`) and converts it to NDJSON (`NDJSONProcessor`) for embedding processing.
4. **Vector DB Ingestion** (`vector_db_ingestion_agent`): Ingests the processed records into PostgreSQL via `VectorDbPipeline`.

```text
PostgreSQL Setup
    ↓
Codebase Analysis (StaticAnalyzerAgent — 9 analyzers + optional deep adapters)
    ↓
Flatten & NDJSON
    ↓
Vector DB Ingestion
```

### Exclusive LLM Mode (`--llm-exclusive`)

Bypasses the LangGraph workflow entirely. Runs `CodebaseLLMAgent` for per-file semantic analysis producing `detailed_code_review.xlsx`. When combined with `--enable-adapters`, deep adapter results (complexity, security, dead code, call graph, function metrics) are merged as `static_*` tabs in the same Excel file.

```text
[Optional] Deep Static Adapters (Lizard, Flawfinder, CCLS)
    ↓
CodebaseLLMAgent (per-file LLM analysis)
    ↓
detailed_code_review.xlsx (LLM tabs + static_ adapter tabs)
```

### Deep Static Analysis Adapters (`--enable-adapters`)

When enabled, the following adapters run using real tooling instead of regex heuristics:

| Adapter | Backend | Capabilities |
| :--- | :--- | :--- |
| `ASTComplexityAdapter` | Lizard | Real cyclomatic complexity, nesting depth, parameter counts |
| `SecurityAdapter` | Flawfinder | CWE-mapped vulnerability scanning with severity levels |
| `DeadCodeAdapter` | CCLS/libclang | BFS reachability analysis from entry points |
| `CallGraphAdapter` | CCLS/libclang | Fan-in/fan-out, cycle detection, max call depth |
| `FunctionMetricsAdapter` | CCLS/libclang | Function body lines, parameters, templates, virtuals |
| `ExcelReportAdapter` | ExcelWriter | Generates `static_*` prefixed tabs in Excel output |

All adapters inherit from `BaseStaticAdapter` and degrade gracefully when their underlying tool is unavailable.

### Project Layout

```text
.
├── main.py                             # Entry point & LangGraph workflow
├── fixer_workflow.py                   # Human-in-the-loop repair workflow
├── global_config.yaml                  # Hierarchical YAML configuration
├── requirements.txt                    # Python dependencies
├── agents/
│   ├── static_analyzer_agent.py        # Unified 7-phase static analyzer
│   ├── codebase_llm_agent.py           # LLM-exclusive per-file code reviewer
│   ├── codebase_fixer_agent.py         # Agentic code repair agent
│   ├── codebase_analysis_chat_agent.py # Interactive chat analysis agent
│   ├── adapters/                       # Deep static analysis adapters
│   │   ├── base_adapter.py             #   ABC base class
│   │   ├── ast_complexity_adapter.py   #   Lizard integration
│   │   ├── security_adapter.py         #   Flawfinder integration
│   │   ├── dead_code_adapter.py        #   CCLS dead code detection
│   │   ├── call_graph_adapter.py       #   CCLS call graph analysis
│   │   ├── function_metrics_adapter.py #   CCLS function metrics
│   │   └── excel_report_adapter.py     #   static_ Excel tab generator
│   ├── analyzers/                      # 9 regex-based health analyzers
│   │   ├── base_runtime_analyzer.py    #   ABC base for all analyzers
│   │   ├── complexity_analyzer.py
│   │   ├── security_analyzer.py
│   │   ├── dependency_analyzer.py
│   │   ├── memory_corruption_analyzer.py
│   │   ├── null_pointer_analyzer.py
│   │   ├── potential_deadlock_analyzer.py
│   │   ├── quality_analyzer.py
│   │   ├── maintainability_analyzer.py
│   │   ├── documentation_analyzer.py
│   │   └── test_coverage_analyzer.py
│   ├── core/
│   │   ├── file_processor.py           # File discovery & caching
│   │   └── metrics_calculator.py       # Orchestrates analyzers + adapters
│   ├── prompts/
│   │   └── prompts.py                  # PromptTemplates for LLM agents
│   ├── parsers/
│   │   ├── excel_to_agent_parser.py    # Excel → JSONL directives parser
│   │   ├── healthreport_generator.py   # HTML health report renderer
│   │   └── healthreport_parser.py      # Legacy health report parser
│   ├── visualization/
│   │   └── graph_generator.py          # Dependency graph visualization
│   └── vector_db/
│       └── document_processor.py       # Vector document processing
├── db/
│   ├── postgres_db_setup.py            # PostgreSQL schema setup
│   ├── postgres_api.py                 # PostgreSQL API helpers
│   ├── json_flattner.py                # JSON → flat JSON converter
│   ├── ndjson_processor.py             # NDJSON processor for embeddings
│   ├── ndjson_writer.py                # NDJSON file writer
│   ├── vectordb_pipeline.py            # Vector DB ingestion pipeline
│   └── vectordb_wrapper.py             # Vector DB abstraction wrapper
├── dependency_builder/                 # CCLS / libclang integration
│   ├── ccls_code_navigator.py          # LSP-based code navigation
│   ├── ccls_ingestion.py               # CCLS indexing orchestrator
│   ├── ccls_dependency_builder.py      # Dependency graph builder
│   ├── dependency_service.py           # Dependency resolution service
│   ├── dependency_handler.py           # Dependency processing handler
│   ├── connection_pool.py              # LSP connection pooling
│   ├── config.py                       # DependencyBuilderConfig
│   ├── models.py                       # Data models
│   ├── metrics.py                      # Performance metrics
│   ├── lsp_notification_handlers.py    # CCLS LSP notification handlers
│   ├── exceptions.py                   # Custom exceptions
│   └── utils.py                        # Shared utilities
├── utils/
│   ├── common/
│   │   ├── llm_tools.py                # Multi-provider LLM abstraction
│   │   ├── excel_writer.py             # Excel report generator
│   │   ├── email_reporter.py           # SMTP email reporter
│   │   └── mmdtopdf.py                # Mermaid → PDF converter
│   ├── parsers/
│   │   ├── env_parser.py               # .env / environment config
│   │   └── global_config_parser.py     # YAML GlobalConfig parser
│   └── prompts/
│       └── prompts.py                  # Utility prompt helpers
├── prompts/
│   └── codebase_analysis_prompt.py     # LLM analysis prompt template
└── ui/
    ├── streamlit_app.py                # Streamlit dashboard
    ├── streamlit_tools.py              # Custom Streamlit helpers
    └── launch_streamlit.py             # Streamlit launcher
```

---

## Installation & Setup

### 1. System Prerequisites

**Python 3.12+** is recommended.

**Install CCLS** (required for `dependency_builder` and deep adapters):

```bash
# macOS
brew install ccls

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ccls

# snap alternative
sudo snap install ccls --classic

# Windows (via Chocolatey)
choco install ccls
```

**Install PostgreSQL** (required for vector DB pipeline):

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql postgresql-client postgresql-16-pgvector
sudo systemctl start postgresql
```

### 2. Database Initialization

```bash
sudo -u postgres psql
```

```sql
-- 1. Create the application user
CREATE USER codebase_analytics_user WITH PASSWORD 'postgres';

-- 2. Create the database
CREATE DATABASE codebase_analytics_db OWNER codebase_analytics_user;

-- 3. Connect to the database
\c codebase_analytics_db

-- 4. Install Vector Extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 5. Grant Permissions
GRANT ALL PRIVILEGES ON DATABASE codebase_analytics_db TO codebase_analytics_user;
GRANT USAGE ON SCHEMA public TO codebase_analytics_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO codebase_analytics_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO codebase_analytics_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO codebase_analytics_user;
```

### 3. Python Environment Setup

```bash
# Create and activate virtual environment
python3.12 -m venv ~/venv/qgenie_py312
source ~/venv/qgenie_py312/bin/activate
python --version
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

For advanced CCLS builds from source (optional):

```bash
sudo apt install clang-14 libclang-14-dev llvm-14-dev
export CC=clang-14
export CXX=clang++-14
git clone --depth=1 --recursive https://github.com/MaskRay/ccls
cd ccls
cmake -S . -B Release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/lib/llvm-14
cmake --build Release --target install
```

### 5. Configuration

Copy and customize the global configuration:

```bash
cp global_config.yaml.example global_config.yaml
# OR edit global_config.yaml directly
```

Set your LLM provider in `global_config.yaml`:

```yaml
llm:
  model: anthropic::claude-sonnet-4-20250514   # Anthropic Claude
  # model: qgenie::qwen2.5-14b-1m              # QGenie
  # model: vertexai::gemini-2.5-pro             # Google Vertex AI
  # model: azure::gpt-4.1                       # Azure OpenAI
```

Set API keys via environment or `.env`:

```bash
cp env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-...
#   QGENIE_API_KEY=...
#   POSTGRES_PASSWORD=...
```

---

## Usage

### CLI Reference

| Flag | Description |
| :--- | :--- |
| `--codebase-path PATH` | Path to the C/C++ codebase to analyze |
| `-d, --out-dir DIR` | Output directory for generated files (default: `./out`) |
| `--config-file PATH` | Path to `global_config.yaml` (default: auto-detected) |
| `--use-llm` | Enable LLM enrichment in StaticAnalyzerAgent (health report mode) |
| `--llm-exclusive` | Use CodebaseLLMAgent exclusively for Excel report (skips health pipeline) |
| `--enable-adapters` | Run deep static analysis adapters (Lizard, Flawfinder, CCLS) |
| `--use-ccls` | Enable CCLS dependency services for CodebaseLLMAgent |
| `--file-to-fix FILE` | Analyze a specific file (relative to codebase path) |
| `--llm-model MODEL` | LLM model in `provider::model` format (overrides config) |
| `--llm-api-key KEY` | API key for the LLM provider |
| `--llm-max-tokens N` | Max tokens per LLM request (default: 15000) |
| `--llm-temperature F` | LLM temperature (default: 0.1) |
| `--max-files N` | Max files to analyze (default: 2000) |
| `--batch-size N` | Files per analysis batch (default: 25) |
| `--exclude-dirs D [D]` | Directories to exclude from analysis |
| `--exclude-globs G [G]` | Glob patterns to exclude (e.g., `*.test.cpp`) |
| `--enable-vector-db` | Enable vector DB ingestion pipeline |
| `--vector-chunk-size N` | Chunk size for vector embeddings (default: 4000) |
| `--vector-overlap-size N` | Overlap between chunks (default: 200) |
| `--vector-include-code` | Include source code in vector embeddings (default: on) |
| `--enable-chatbot-optimization` | Enable chatbot-optimized vector processing |
| `--generate-report` | Generate HTML health report from healthreport.json |
| `--generate-visualizations` | Generate dependency graph visualizations |
| `--generate-pdfs` | Generate PDF outputs from Mermaid diagrams |
| `--max-edges N` | Max edges in graph visualizations (default: 500) |
| `--health-report-path PATH` | Override path for healthreport.json output |
| `--flat-json-path PATH` | Override path for flattened JSON output |
| `--ndjson-path PATH` | Override path for NDJSON output |
| `--force-reanalysis` | Force re-analysis ignoring cached results |
| `--memory-limit MB` | Memory limit in MB (default: 3000) |
| `--enable-memory-monitoring` | Enable real-time memory monitoring (default: on) |
| `-v, --verbose` | Verbose logging |
| `-D, --debug` | Debug logging |
| `--quiet` | Suppress non-error output |

### Standard Analysis (Health Report Pipeline)

```bash
# Basic static analysis (fast, regex-based)
python main.py --codebase-path /path/to/cpp/project

# With LLM enrichment
python main.py --codebase-path /path/to/cpp/project --use-llm

# With deep static adapters (Lizard + Flawfinder + CCLS)
python main.py --codebase-path /path/to/cpp/project --enable-adapters

# Full pipeline with vector DB
python main.py --codebase-path /path/to/cpp/project \
  --use-llm --enable-adapters --enable-vector-db --generate-report
```

### Exclusive LLM Analysis (Direct Excel Report)

This mode skips the LangGraph health pipeline and generates `detailed_code_review.xlsx` directly.

```bash
# LLM-only analysis
python main.py --llm-exclusive --codebase-path /path/to/cpp/project

# LLM + deep adapters (static_ tabs merged into same Excel)
python main.py --llm-exclusive --enable-adapters --codebase-path /path/to/cpp/project

# With CCLS dependency context
python main.py --llm-exclusive --enable-adapters --use-ccls \
  --codebase-path /path/to/cpp/project

# Targeted single-file analysis
python main.py --llm-exclusive --use-ccls \
  --file-to-fix "src/module/component.cpp" \
  --codebase-path /path/to/cpp/project
```

### LLM Provider Selection

```bash
# Anthropic Claude
python main.py --llm-exclusive --llm-model "anthropic::claude-sonnet-4-20250514" \
  --codebase-path /path/to/project

# Google Vertex AI
python main.py --llm-exclusive --llm-model "vertexai::gemini-2.5-pro" \
  --codebase-path /path/to/project

# Azure OpenAI
python main.py --llm-exclusive --llm-model "azure::gpt-4.1" \
  --codebase-path /path/to/project
```

### Streamlit Dashboard

```bash
# Requirement: Run the pipeline with --enable-vector-db first
python -m streamlit run ui/streamlit_app.py --server.port 8502
```

Access at: `http://localhost:8502`

---

## Agentic Code Repair (Human-in-the-Loop)

The pipeline includes a **CodebaseFixerAgent** that closes the loop between analysis and remediation.

### Workflow

1. **Analyze**: Run `main.py` to generate `detailed_code_review.xlsx`.
2. **Human Review**: Open the Excel, review High/Critical issues, add feedback and constraints.
3. **Execute Fixes**: Run `fixer_workflow.py` to apply LLM-guided fixes.

### Fixer Commands

```bash
# Single-step (parse + fix)
python fixer_workflow.py --excel detailed_code_review.xlsx --codebase-path /path/to/project

# Parse Excel to JSONL only
python fixer_workflow.py --step parse --excel detailed_code_review.xlsx

# Run the fixer agent only
python fixer_workflow.py --step fix --codebase-path /path/to/project
```

### Feedback & Constraints Columns

**Feedback column** — controls the action:

| User Input | Effect |
| :--- | :--- |
| *(Empty)* | **Approve.** Apply `Fixed_Code` as suggested. |
| `Skip` / `Ignore` / `No Fix` | **Reject.** File untouched (false positive). |
| `Approved` / `LGTM` | **Approve.** Explicit confirmation. |
| `Modify` / `Update` / `Retry` | **Custom fix.** Re-generate using Constraints column. |

**Constraints column** — provides technical guardrails for custom fixes, for example: "Use `std::array` instead of `std::vector`", "Follow C++98 only", "Wrap in `std::lock_guard`", etc.

### Fixer Features

- **Holistic Refactoring**: Fixes multiple issues in a single file simultaneously for consistency.
- **Smart Backups**: Creates a mirror in `out/shelved_backups` before modifying any file.
- **Safety Gates**: Checks for file size anomalies and LLM failures before overwriting.
- **Audit Reporting**: Produces `final_execution_audit.xlsx` with color-coded status (FIXED, SKIPPED, FAILED).

---

## Configuration Reference

### global_config.yaml

The `global_config.yaml` file provides hierarchical, typed configuration with `${ENV_VAR}` override support. Key sections:

| Section | Purpose |
| :--- | :--- |
| `paths` | Input/output directories, prompt file paths |
| `llm` | Provider, model, API keys, token limits, temperature |
| `embeddings` | Vector embedding model selection |
| `database` | PostgreSQL connection, PGVector collection settings |
| `email` | SMTP report delivery configuration |
| `dependency_builder` | CCLS executable, timeouts, BFS depth, connection pool |
| `excel` | Report styling (colors, column widths, freeze/filter) |
| `mermaid` | Diagram rendering configuration |
| `logging` | Log level, verbose/debug flags |

---

## Troubleshooting

### Memory Optimization

```bash
# Memory-optimized run for large codebases
python main.py --codebase-path ./codebase \
  --max-files 1000 --batch-size 50 --memory-limit 3000

# Debug mode with monitoring
python main.py --debug --enable-memory-monitoring --max-files 500
```

### Database Maintenance

```bash
psql -h localhost -U codebase_analytics_user -d codebase_analytics_db
```

```sql
-- Check embeddings
SELECT document, cmetadata FROM langchain_pg_embedding LIMIT 5;

-- Clear all vector data
DELETE FROM langchain_pg_embedding;
DELETE FROM langchain_pg_collection;
```

### Common Issues

- **`ModuleNotFoundError: No module named 'networkx'`**: Run `pip install -r requirements.txt` to install all dependencies.
- **Adapters show "tool not available"**: Install optional tools — `pip install lizard flawfinder`. For CCLS adapters, ensure `ccls` is installed and in PATH.
- **CCLS indexing timeout**: Increase `dependency_builder.indexing_timeout_seconds` in `global_config.yaml`.
- **LLM provider errors**: Verify `--llm-model` uses the correct `provider::model` format and that the corresponding API key is set.

---

## Contributing

Contributions are welcome! Please open issues and pull requests for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
