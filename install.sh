#!/usr/bin/env bash
# ============================================================================
# CARE — Codebase Analysis & Repair Engine for Silicon Design
# One-step installer for macOS, Linux, and Windows (WSL)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# What this script does:
#   1. Detects OS and package manager
#   2. Installs Python 3.12+ (if needed)
#   3. Creates a virtual environment (.venv)
#   4. Installs all Python dependencies
#   5. Installs HDL tools: Verible, Verilator, Yosys, Icarus Verilog
#   6. Installs optional tools: Pandoc, Mermaid CLI (mmdc)
#   7. Sets up .env from env.example (if not present)
#   8. Validates the installation
#   9. Prints launch instructions
#
# Environment variable overrides:
#   CARE_PYTHON=python3.12    Override Python binary
#   CARE_SKIP_HDL=1           Skip HDL tool installation
#   CARE_SKIP_DB=1            Skip PostgreSQL/pgvector setup
#   CARE_SKIP_OPTIONAL=1      Skip optional tools (Pandoc, mmdc)
#   CARE_VENV_DIR=.venv       Override virtual environment path
# ============================================================================

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Logging ─────────────────────────────────────────────────────────────────
info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
err()     { echo -e "${RED}[✗]${NC} $*"; }
step()    { echo -e "\n${CYAN}${BOLD}── $* ──${NC}"; }
substep() { echo -e "  ${BLUE}→${NC} $*"; }

# ── Project root ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable ────────────────────────────────────────────────────────────
VENV_DIR="${CARE_VENV_DIR:-.venv}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=9

# ============================================================================
# Banner
# ============================================================================
echo -e "${CYAN}${BOLD}"
cat << 'BANNER'

   ██████╗ █████╗ ██████╗ ███████╗
  ██╔════╝██╔══██╗██╔══██╗██╔════╝
  ██║     ███████║██████╔╝█████╗
  ██║     ██╔══██║██╔══██╗██╔══╝
  ╚██████╗██║  ██║██║  ██║███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
  Codebase Analysis & Repair Engine
  Silicon Design HDL Framework

BANNER
echo -e "${NC}"

# ============================================================================
# Step 0: Detect OS & Package Manager
# ============================================================================
step "Detecting environment"

OS_TYPE="unknown"
PKG_MANAGER="unknown"
IS_WSL=false

case "$(uname -s)" in
    Darwin)
        OS_TYPE="macos"
        if command -v brew &>/dev/null; then
            PKG_MANAGER="brew"
        else
            err "Homebrew is required on macOS."
            err "Install it: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        ;;
    Linux)
        OS_TYPE="linux"
        # Check for WSL
        if grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then
            IS_WSL=true
            info "Windows Subsystem for Linux (WSL) detected"
        fi
        if command -v apt-get &>/dev/null; then
            PKG_MANAGER="apt"
        elif command -v dnf &>/dev/null; then
            PKG_MANAGER="dnf"
        elif command -v yum &>/dev/null; then
            PKG_MANAGER="yum"
        elif command -v pacman &>/dev/null; then
            PKG_MANAGER="pacman"
        else
            warn "No supported package manager found. Some tools may need manual installation."
            PKG_MANAGER="none"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        err "Native Windows detected. Please use WSL (Windows Subsystem for Linux)."
        err "Install WSL:  wsl --install"
        err "Then re-run this script inside WSL."
        exit 1
        ;;
    *)
        warn "Unknown OS: $(uname -s). Proceeding with best effort..."
        OS_TYPE="linux"
        PKG_MANAGER="none"
        ;;
esac

info "OS: ${OS_TYPE}$(${IS_WSL} && echo ' (WSL)' || echo '')  |  Package manager: ${PKG_MANAGER}"

# ── Helper: install a system package ────────────────────────────────────────
pkg_install() {
    local pkg_name="$1"
    local brew_name="${2:-$1}"
    local apt_name="${3:-$1}"

    substep "Installing ${pkg_name}..."
    case "$PKG_MANAGER" in
        brew)   brew install "$brew_name" 2>/dev/null || true ;;
        apt)    sudo apt-get install -y -qq "$apt_name" 2>/dev/null || true ;;
        dnf)    sudo dnf install -y "$apt_name" 2>/dev/null || true ;;
        yum)    sudo yum install -y "$apt_name" 2>/dev/null || true ;;
        pacman) sudo pacman -S --noconfirm "$apt_name" 2>/dev/null || true ;;
        *)      warn "Cannot auto-install ${pkg_name}. Please install manually." ;;
    esac
}

# ── Helper: check if command exists ─────────────────────────────────────────
has_cmd() { command -v "$1" &>/dev/null; }

# ============================================================================
# Step 1: Python
# ============================================================================
step "Checking Python"

find_python() {
    # Check override first
    if [[ -n "${CARE_PYTHON:-}" ]] && has_cmd "$CARE_PYTHON"; then
        echo "$CARE_PYTHON"
        return
    fi

    # Try common Python names
    for py in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if has_cmd "$py"; then
            local ver
            ver="$($py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
            local major minor
            major="${ver%%.*}"
            minor="${ver#*.}"
            if [[ "$major" -ge "$MIN_PYTHON_MAJOR" && "$minor" -ge "$MIN_PYTHON_MINOR" ]]; then
                echo "$py"
                return
            fi
        fi
    done
    echo ""
}

PYTHON_BIN="$(find_python)"

if [[ -z "$PYTHON_BIN" ]]; then
    warn "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ not found. Installing..."
    case "$PKG_MANAGER" in
        brew)   brew install python@3.12 2>/dev/null || true ;;
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip 2>/dev/null || \
            sudo apt-get install -y -qq python3 python3-venv python3-pip 2>/dev/null || true
            ;;
        dnf)    sudo dnf install -y python3.12 python3-pip 2>/dev/null || \
                sudo dnf install -y python3 python3-pip 2>/dev/null || true ;;
        pacman) sudo pacman -S --noconfirm python python-pip 2>/dev/null || true ;;
        *)      err "Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ manually."; exit 1 ;;
    esac
    PYTHON_BIN="$(find_python)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    err "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required but could not be found or installed."
    exit 1
fi

PY_VERSION="$($PYTHON_BIN --version 2>&1)"
info "Using: ${PY_VERSION} ($(which $PYTHON_BIN))"

# ============================================================================
# Step 2: Virtual Environment
# ============================================================================
step "Setting up virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at ${VENV_DIR}"
else
    substep "Creating virtual environment..."
    $PYTHON_BIN -m venv "$VENV_DIR"
    info "Created virtual environment at ${VENV_DIR}"
fi

# Activate
source "${VENV_DIR}/bin/activate"
info "Activated virtual environment"

# Upgrade pip
substep "Upgrading pip..."
pip install --upgrade pip --quiet 2>/dev/null

# ============================================================================
# Step 3: Python Dependencies
# ============================================================================
step "Installing Python dependencies"

if [[ -f "requirements.txt" ]]; then
    substep "Installing from requirements.txt..."
    pip install -r requirements.txt --quiet 2>&1 | tail -5 || {
        warn "Some packages failed. Trying individually..."
        while IFS= read -r line; do
            # Skip comments and blank lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// /}" ]] && continue
            pip install "$line" --quiet 2>/dev/null || warn "Failed to install: $line"
        done < requirements.txt
    }
    info "Python dependencies installed"
else
    err "requirements.txt not found!"
    exit 1
fi

# ============================================================================
# Step 4: HDL Tools
# ============================================================================
if [[ "${CARE_SKIP_HDL:-0}" != "1" ]]; then
    step "Installing HDL tools"

    # ── Verible (SystemVerilog parser & linter) ─────────────────────────────
    if has_cmd verible-verilog-syntax; then
        info "Verible already installed: $(verible-verilog-syntax --version 2>&1 | head -1)"
    else
        substep "Installing Verible..."
        case "$PKG_MANAGER" in
            brew) brew install verible 2>/dev/null || true ;;
            apt)
                # Try apt first, then GitHub release
                if ! sudo apt-get install -y -qq verible 2>/dev/null; then
                    substep "Fetching Verible from GitHub releases..."
                    VERIBLE_VER="v0.0-3824-g74aafdb6"
                    ARCH="$(uname -m)"
                    case "$ARCH" in
                        x86_64|amd64) VERIBLE_ARCH="x86_64" ;;
                        aarch64|arm64) VERIBLE_ARCH="aarch64" ;;
                        *) warn "Unsupported arch for Verible: $ARCH"; VERIBLE_ARCH="" ;;
                    esac
                    if [[ -n "$VERIBLE_ARCH" ]]; then
                        VERIBLE_URL="https://github.com/chipsalliance/verible/releases/download/${VERIBLE_VER}/verible-${VERIBLE_VER}-linux-static-${VERIBLE_ARCH}.tar.gz"
                        curl -fsSL "$VERIBLE_URL" -o /tmp/verible.tar.gz 2>/dev/null && \
                        sudo tar -xzf /tmp/verible.tar.gz -C /usr/local --strip-components=1 2>/dev/null && \
                        rm -f /tmp/verible.tar.gz && \
                        info "Verible installed from GitHub" || \
                        warn "Failed to install Verible. Install manually from: https://github.com/chipsalliance/verible/releases"
                    fi
                fi
                ;;
            *) warn "Please install Verible manually: https://github.com/chipsalliance/verible/releases" ;;
        esac
    fi

    # ── Verilator ───────────────────────────────────────────────────────────
    if has_cmd verilator; then
        info "Verilator already installed: $(verilator --version 2>&1 | head -1)"
    else
        pkg_install "Verilator" "verilator" "verilator"
    fi

    # ── Yosys (open-source synthesis) ───────────────────────────────────────
    if has_cmd yosys; then
        info "Yosys already installed: $(yosys --version 2>&1 | head -1)"
    else
        pkg_install "Yosys" "yosys" "yosys"
    fi

    # ── Icarus Verilog (simulation) ─────────────────────────────────────────
    if has_cmd iverilog; then
        info "Icarus Verilog already installed: $(iverilog -V 2>&1 | head -1)"
    else
        pkg_install "Icarus Verilog" "icarus-verilog" "iverilog"
    fi

    # Summary
    echo ""
    substep "HDL Tool Summary:"
    for tool in verible-verilog-syntax verilator yosys iverilog; do
        if has_cmd "$tool"; then
            echo -e "    ${GREEN}✓${NC} $tool"
        else
            echo -e "    ${YELLOW}○${NC} $tool (not found — optional)"
        fi
    done
else
    step "Skipping HDL tools (CARE_SKIP_HDL=1)"
fi

# ============================================================================
# Step 5: Optional Tools
# ============================================================================
if [[ "${CARE_SKIP_OPTIONAL:-0}" != "1" ]]; then
    step "Installing optional tools"

    # ── Pandoc (document conversion) ────────────────────────────────────────
    if has_cmd pandoc; then
        info "Pandoc already installed"
    else
        pkg_install "Pandoc" "pandoc" "pandoc"
    fi

    # ── Node.js + Mermaid CLI (diagram rendering) ───────────────────────────
    if has_cmd mmdc; then
        info "Mermaid CLI already installed"
    else
        if has_cmd npm; then
            substep "Installing Mermaid CLI (mmdc)..."
            npm install -g @mermaid-js/mermaid-cli 2>/dev/null || warn "Failed to install mmdc"
        elif has_cmd node; then
            warn "npm not found. Install Node.js to get Mermaid CLI support."
        else
            substep "Installing Node.js..."
            case "$PKG_MANAGER" in
                brew)   brew install node 2>/dev/null || true ;;
                apt)    sudo apt-get install -y -qq nodejs npm 2>/dev/null || true ;;
                dnf)    sudo dnf install -y nodejs npm 2>/dev/null || true ;;
                pacman) sudo pacman -S --noconfirm nodejs npm 2>/dev/null || true ;;
                *)      warn "Please install Node.js manually for Mermaid support." ;;
            esac
            if has_cmd npm; then
                npm install -g @mermaid-js/mermaid-cli 2>/dev/null || warn "Failed to install mmdc"
            fi
        fi
    fi
else
    step "Skipping optional tools (CARE_SKIP_OPTIONAL=1)"
fi

# ============================================================================
# Step 6: PostgreSQL & pgvector (optional)
# ============================================================================
if [[ "${CARE_SKIP_DB:-0}" != "1" ]]; then
    step "Database setup"
    if [[ -f "bootstrap_db.sh" ]]; then
        info "PostgreSQL bootstrap script found (bootstrap_db.sh)"
        info "Run it separately when you're ready to set up the database:"
        echo -e "    ${CYAN}sudo ./bootstrap_db.sh${NC}"
        echo ""
        info "DB credentials are configured in global_config.yaml (database: section)"
        info "The database is optional — core analysis works without it."
        info "It's needed for: vector DB search, telemetry, and HITL feedback."
    else
        warn "bootstrap_db.sh not found. Database setup must be done manually."
    fi
else
    step "Skipping database setup (CARE_SKIP_DB=1)"
fi

# ============================================================================
# Step 7: Environment Configuration
# ============================================================================
step "Setting up environment"

if [[ ! -f ".env" ]]; then
    if [[ -f "env.example" ]]; then
        cp env.example .env
        info "Created .env from env.example"
        warn "Edit .env to add your API keys (LLM_API_KEY, etc.)"
    else
        warn "No env.example found. Create a .env file with your API keys."
    fi
else
    info ".env file already exists"
fi

# Create output directory
mkdir -p out
info "Output directory ready (./out)"

# ============================================================================
# Step 8: Validation
# ============================================================================
step "Validating installation"

ERRORS=0

# Python packages
substep "Checking Python packages..."
for pkg in streamlit pandas anthropic networkx openpyxl rich; do
    if python -c "import $pkg" 2>/dev/null; then
        echo -e "    ${GREEN}✓${NC} $pkg"
    else
        echo -e "    ${RED}✗${NC} $pkg"
        ERRORS=$((ERRORS + 1))
    fi
done

# Core files
substep "Checking project files..."
for f in main.py ui/app.py ui/launch.py ui/streamlit_tools.py global_config.yaml requirements.txt; do
    if [[ -f "$f" ]]; then
        echo -e "    ${GREEN}✓${NC} $f"
    else
        echo -e "    ${RED}✗${NC} $f (missing)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Syntax validation
substep "Validating Python syntax..."
for f in ui/app.py ui/streamlit_tools.py ui/launch.py main.py; do
    if [[ -f "$f" ]]; then
        if python -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
            echo -e "    ${GREEN}✓${NC} $f"
        else
            echo -e "    ${RED}✗${NC} $f (syntax error)"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

# ============================================================================
# Step 9: Summary & Launch Instructions
# ============================================================================
echo ""
echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"

if [[ "$ERRORS" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  ✓ CARE installation complete!${NC}"
else
    echo -e "${YELLOW}${BOLD}  ! CARE installed with ${ERRORS} warning(s)${NC}"
fi

echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo ""
echo -e "    ${CYAN}# 1. Activate the environment${NC}"
echo -e "    source ${VENV_DIR}/bin/activate"
echo ""
echo -e "    ${CYAN}# 2. Add your API key${NC}"
echo -e "    export LLM_API_KEY=\"sk-...\""
echo -e "    ${CYAN}# (or edit .env)${NC}"
echo ""
echo -e "    ${CYAN}# 3. Launch the dashboard${NC}"
echo -e "    ./launch.sh"
echo -e "    ${CYAN}# or: python ui/launch.py${NC}"
echo ""
echo -e "    ${CYAN}# 4. Open the website${NC}"
echo -e "    open index.html    ${CYAN}# macOS${NC}"
echo -e "    xdg-open index.html ${CYAN}# Linux${NC}"
echo ""
echo -e "    ${CYAN}# 5. Run CLI analysis${NC}"
echo -e "    python main.py --source ./rtl --out ./out"
echo ""
echo -e "  ${BOLD}Optional:${NC}"
echo -e "    sudo ./bootstrap_db.sh    ${CYAN}# Set up PostgreSQL${NC}"
echo -e "    python main.py --enable-vector-db  ${CYAN}# Enable vector search${NC}"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
