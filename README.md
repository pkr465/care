# C/C++ Codebase Health Analysis & Vector DB Pipeline

`main.py` is the main entry point for a **multi-stage C/C++ codebase health analysis pipeline**. It is designed to analyze C/C++ codebases and produce rich health scores, structural metadata, and embeddings suitable for RAG (Retrieval-Augmented Generation) applications.

## Key Capabilities

1. **Analyze C/C++ Codebases**: Uses an incremental static analyzer (fast) or an LLM-powered `CodebaseAnalysisAgent` (deep insights).
2. **Health Reporting**: Produces a canonical `healthreport.json` with metrics and summaries.
3. **Data Flattening**: Converts reports to JSON and NDJSON formats for embedding.
4. **Vector DB Ingestion**: Ingests data into a **PostgreSQL** vector database.
5. **Visualization**: Generates an HTML health report and provides a Streamlit UI.

This pipeline supports architecture reviews, technical debt tracking, CI/CD health gates, and code-centric semantic search.

---

## Architecture & Workflow

The workflow is orchestrated using **LangGraph** and consists of four main agents:

1. **PostgreSQL Setup** (`postgres_db_setup_agent`):
   - Sets up the schema and tables for vector storage.
2. **Codebase Analysis** (`codebase_analysis_agent`):
   - Validates the codebase structure.
   - Runs either the `CodebaseAnalysisAgent` (LLM-based) or `IncrementalCodebaseAnalyzer` (Static).
   - Generates `healthreport.json`.
3. **Flatten & NDJSON** (`flatten_and_ndjson_agent`):
   - Flattens the report (`JsonFlattener`) and converts it to NDJSON (`NDJSONWriter`) for embedding processing.
4. **Vector DB Ingestion** (`vector_db_ingestion_agent`):
   - Ingests the processed records into PostgreSQL via `VectorDbPipeline`.

### Execution Flow

```text
PostgreSQL Setup
    ↓
Codebase Analysis
    ↓
Flatten & NDJSON
    ↓
Vector DB Ingestion
```

### Project Layout

The expected project structure:

```text
.
├── main.py
├── agents/
│   ├── codebase_analysis_agent.py    # LLM-powered codebase analysis
│   ├── incremental_analyzer.py       # Incremental static analyzer
│   ├── parsers/
│   │   ├── healthreport_generator.py # Optional HTML report generator
│   │   └── healthreport_parser.py    # Optional legacy parser
│   └── vector_db/
│       └── document_processor.py     # Optional legacy vector processing
├── db/
│   ├── postgres_db_setup.py          # PostgreSQL setup logic (PostgresDbSetup)
│   ├── json_flattner.py              # JsonFlattener
│   ├── ndjson_processor.py           # NDJSONProcessor
│   ├── ndjson_writer.py              # NDJSONWriter
│   └── vectordb_pipeline.py          # VectorDbPipeline
└── utils/
    └── parsers/
        └── env_parser.py             # EnvConfig
```

---

## Installation & Setup

### 1. System Prerequisites
*Use `sudo` if you encounter permission issues.*

**Install Miniconda (Optional):**
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Accept license, default path, say "yes" to conda init
source ~/.bashrc  # or ~/.zshrc
```

**Install PostgreSQL (Linux):**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql postgresql-client postgresql-16-pgvector

# Start the service
sudo systemctl start postgresql
sudo systemctl status postgresql
```

### 2. Database Initialization
You must create the database and user before running the pipeline.

**Become the postgres superuser:**
```bash
sudo -u postgres psql
```

**Run the following SQL commands:**
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
Create a virtual environment (Python 3.12 recommended).

```bash
# Ensure no venv is active
deactivate 2>/dev/null || true

# Create venv using Python 3.12
/usr/bin/python3.12 -m venv ~/venv/codebase_analysis_py312

# Activate it
source ~/venv/codebase_analysis_py312/bin/activate

# Verify version
python --version
```

### 4. Install Dependencies
Run these commands **in order**.

```bash
pip install psutil
pip install rich
pip install numpy psycopg2-binary
pip install sqlalchemy langgraph langchain-core

# Qualcomm Internal QGenie SDK
pip install qgenie-sdk -i https://devpi.qualcomm.com/qcom/dev/+simple --trusted-host devpi.qualcomm.com
pip install "qgenie-sdk[integrations]" -i https://devpi.qualcomm.com/qcom/dev/+simple --trusted-host devpi.qualcomm.com

pip install langchain==0.0.350
pip install langchain-postgres
pip install networkx
pip install dotenv
pip install streamlit
pip install matplotlib
pip install pylspclient

#For devcompute

pip install clang==14.0.6
# Optional (not required for standard run):
# pip install "langgraph==0.0.40" "langchain-core==0.1.0"
```

### 5. Configuration
```bash
cp env.example .env
# Edit .env and add your QGENIE_API_KEY
```
** --for mac **
``` sh
brew install ccls
```

** -- on Linux **
``` sh
sudo apt-get update
sudo apt-get install -y ccls
```

-- alternatively
``` sh
sudo snap install ccls --classic
```

**-- windows 
install Chocolatey (if you don't have it):
Open an elevated Command Prompt or PowerShell session and follow the installation instructions on the Chocolatey website. **

``` sh
choco install ccls
```

** only for advanced users **
```sh
sudo apt install clang-14 libclang-14-dev llvm-14-dev clang-uml


- set env variable 
```sh
export CC=clang-14
export CXX=clang++-14
```
*Note: Set it in path also*
- clone ccls repo
```sh
cd ..
git clone --depth=1 --recursive https://github.com/MaskRay/ccls
```
- build ccls repo
```sh
cd ccls
cmake -S. -BRelease -DCMAKE_BUILD_TYPE=Release     -DCMAKE_PREFIX_PATH=/usr/lib/llvm-7     -DLLVM_INCLUDE_DIR=/usr/lib/llvm-7/include     -DLLVM_BUILD_INCLUDE_DIR=/usr/include/llvm-7/


rm -rf Release

cmake -S . -B Release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$(brew --prefix llvm) \
  -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
  -DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
  -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix llvm)/lib -lc++" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix llvm)/lib -lc++" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix llvm)/lib -lc++"

```
- install ccls : This will create the ccls binary in the Release folder. You can check it works by running ./Release/ccls --version.
```sh
sudo cmake --build Release --target install
```
---

## Usage

### Run the Pipeline
**Standard Analysis (with LLM Report):**
```bash
python main.py --use-llm --generate-report
```

### Exclusive LLM Analysis (Direct Excel Report): This mode skips the vector DB and health metrics pipeline, focusing solely on generating a detailed detailed_code_review.xlsx.
```bash
python main.py --llm-exclusive --codebase-path /path/to/cpp/project
```


**Targeted Analysis (CCLS Mode): Perform deep dependency analysis on a specific file using the CCLS Language Server. Useful for debugging specific components or preparing them for refactoring.**
```bash
python main.py --use-ccls --file-to-fix "<relative path to the file>"

Example:
python main.py --use-ccls --file-to-fix "prplMesh/controller/src/beerocks/master/son_actions.cpp"
```

**Full Pipeline (Vector DB + Chatbot Optimization):**
```bash
python main.py --enable-vector-db --enable-chatbot-optimization --force-reanalysis
```

### Run the UI
The Streamlit frontend provides a visual interface for the data.
*Requirement: Run the pipeline with `--enable-vector-db` first.*

```bash
python -m streamlit run ui/streamlit_app.py --server.port 8502
```
Access the UI at: `http://localhost:8502`

---

# Agentic Code Repair (Human-in-the-Loop)

The pipeline now includes a **"Holistic Refactor" Agent** (`CodebaseFixerAgent`) that closes the loop between analysis and remediation.

## Workflow Steps

1. **Analyze**: Run `main.py` to generate `detailed_code_review.xlsx`.
2. **Human Review**:
   - Open the Excel report.
   - Review High/Critical issues.
   - Provide **Feedback** (context) or **Constraints** (e.g., "Use smart pointers") in the respective columns.
   - (Optional) Mark rows to `SKIP` if they are false positives.
3. **Parse**: Convert the Excel feedback into machine-readable directives.
4. **Execute Fixes**: The agent processes files in batches, sending full context to the LLM to prevent "code drift."


# Guide: Human Feedback & Constraints

**Instructions for Reviewers:** Use the **Feedback** and **Constraints** columns in the Excel report to guide the automated fixing agent.

## Column: Feedback
**Purpose:** To approve, reject, or categorize the issue.

| User Input                    | Effect on Agent                                                                                                          |
| :---------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **(Empty Cell)**              | **Approve.** The agent applies the `Fixed_Code` exactly as originally suggested.                                         |
| `Skip` / `Ignore` / `No Fix`  | **Reject.** The agent touches nothing. Use this for false positives.                                                     |
| `Approved` / `LGTM`           | **Approve.** Explicit confirmation (same as leaving it empty).                                                           |
| `Modify` / `Update` / `Retry` | **Custom Fix.** Signals the agent to ignore the original fix and generate a new one based on the **Constraints** column. |

---

## Column: Constraints
**Purpose:** To provide technical guardrails when the original `Fixed_Code` is insufficient or violates project rules.

### Examples of Valid Constraints:

#### A. Memory & Performance
* "Do not use `std::vector`; use a fixed-size array (`std::array`) to avoid heap allocation."
* "Ensure the fix is `noexcept` compliant."
* "Must use `q_malloc` instead of `malloc` for this module."

#### B. Style & Standards
* "Follow C++98 standard only (no `auto` keyword)."
* "Variable names must be `camelCase`, not `snake_case`."
* "Add a comment marked `// TODO: Refactor` above the fix."

#### C. Thread Safety
* "Wrap this logic in `std::lock_guard<std::mutex> lock(g_mutex);`."
* "Do not use `static` variables here due to reentrancy requirements."

#### D. Business Logic
* "If `value < 0`, log an error instead of throwing an exception."
* "Keep the legacy check for `NULL` before dereferencing."

---

## 3. Agent Logic Workflow

This explains how the **AutoFixAgent** interprets the generated JSONL data:

**1. If `action == "FIX"` (Default/Empty Feedback):**
* The agent trusts the original LLM analysis.
* It applies `suggested_fix` directly to the codebase.

**2. If `action == "SKIP"`:**
* The agent ignores the entry entirely.
* The file is left untouched.

**3. If `action == "FIX_WITH_CONSTRAINTS"`:**
* The agent triggers a **Re-generation Step**.
* It takes the `bad_code_snippet` and the `human_constraints`.
* It calls the LLM again to generate a *new* code fix that adheres to the user's specific rules before applying it.
  
## Scripts & Commands

### 1. Fixer workflow - single step

```bash
python fixer_workflow.py --excel detailed_code_review.xlsx --codebase-path <codebase path>
```

## Individual steps

### 1. Parse Excel to JSONL

```bash
python fixer_workflow.py --step parse --excel detailed_code_review.xlsx
```

### 2. Run the Fixer Agent

```bash
python fixer_workflow.py --step fix --codebase-path ./codebase
```

## Key Features of the Fixer

- **Holistic Refactoring**: Fixes multiple issues in a single file simultaneously to ensure consistency.
- **Smart Backups**: Creates a mirror of the source directory in `out/shelved_backups` before touching any file.
- **Safety Gates**: Checks for file size anomalies and LLM failures before overwriting code.
- **Audit Reporting**: Produces `final_execution_audit.xlsx` with color-coded status (FIXED, SKIPPED, FAILED).


## Troubleshooting & Debugging

### Memory Optimization
For large codebases, use flags to limit memory usage and batch sizes.

```bash
# Memory-optimized run
python main.py \
  --codebase-path ./codebase \
  --max-files 1000 \
  --batch-size 50 \
  --memory-limit 3000 \
  --skip-large-files \
  --force-gc-interval 50

# Debug mode with monitoring
python main.py --debug --enable-memory-monitoring --max-files 500
```

To monitor memory in real-time:
```bash
watch -n 1 'ps aux | grep python | grep codebase'
```

### Database Maintenance
Useful commands for verifying data or resetting tables.

**Connect to DB:**
```bash
psql -h localhost -U codebase_analytics_user -d codebase_analytics_db
```

**SQL Commands:**
```sql
-- Check if embeddings exist
SELECT document, cmetadata FROM langchain_pg_embedding LIMIT 5;

-- Check specific collection
SELECT document, cmetadata, source_file 
FROM langchain_pg_embedding 
WHERE collection_id = 'your-collection-uuid' 
LIMIT 10;

-- Clear all vector data
DELETE FROM langchain_pg_embedding;
DELETE FROM langchain_pg_collection;
```

---

## Contributing
Contributions are welcome! Please open issues and pull requests for any improvements or bug fixes.

## License
This project is licensed under the [MIT License](LICENSE).
