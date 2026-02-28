#!/usr/bin/env bash
# ============================================================================
# CARE — Codebase Analysis & Repair Engine
# Launch script — starts the Streamlit dashboard and optionally opens the
# silicon design website.
#
# Usage:
#   ./launch.sh              # Launch dashboard only
#   ./launch.sh --website    # Launch dashboard + open website
#   ./launch.sh --port 8503  # Custom port
#   ./launch.sh --help       # Show help
#
# Environment:
#   STREAMLIT_PORT=8502      Override default port
#   CARE_NO_BROWSER=1        Don't auto-open browser
# ============================================================================

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# ── Project root ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────
PORT="${STREAMLIT_PORT:-8502}"
OPEN_WEBSITE=false
VENV_DIR=".venv"

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --website|-w)   OPEN_WEBSITE=true; shift ;;
        --port|-p)      PORT="$2"; shift 2 ;;
        --help|-h)
            echo "CARE Launch Script"
            echo ""
            echo "Usage: ./launch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --website, -w    Also open the silicon design website"
            echo "  --port, -p PORT  Set Streamlit port (default: 8502)"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Environment:"
            echo "  STREAMLIT_PORT   Override default port"
            echo "  CARE_NO_BROWSER  Set to 1 to skip auto-opening browser"
            exit 0
            ;;
        *) echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

# ── Banner ──────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   CARE — Codebase Analysis & Repair Engine ║"
echo "  ║   Silicon Design HDL Framework             ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Activate virtual environment ────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    source "${VENV_DIR}/bin/activate"
    echo -e "${GREEN}[✓]${NC} Virtual environment activated"
elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
    echo -e "${GREEN}[✓]${NC} Virtual environment activated"
else
    echo -e "${YELLOW}[!]${NC} No virtual environment found. Using system Python."
    echo -e "    Run ${CYAN}./install.sh${NC} first for a clean setup."
fi

# ── Check Streamlit ─────────────────────────────────────────────────────────
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${RED}[✗]${NC} Streamlit is not installed."
    echo -e "    Run: ${CYAN}pip install -r requirements.txt${NC}"
    exit 1
fi

# ── Check app.py ────────────────────────────────────────────────────────────
APP_PATH="ui/app.py"
if [[ ! -f "$APP_PATH" ]]; then
    echo -e "${RED}[✗]${NC} Cannot find ${APP_PATH}. Run from the project root."
    exit 1
fi

# ── Get local IP for network access ─────────────────────────────────────────
LOCAL_IP="$(python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('10.255.255.255', 1))
    print(s.getsockname()[0])
except: print('localhost')
finally: s.close()
" 2>/dev/null || echo "localhost")"

# ── Print access info ──────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Dashboard URLs:${NC}"
echo -e "    Local:   ${CYAN}http://localhost:${PORT}${NC}"
echo -e "    Network: ${CYAN}http://${LOCAL_IP}:${PORT}${NC}"
echo ""

# ── Open website in browser ─────────────────────────────────────────────────
if [[ "$OPEN_WEBSITE" == "true" && -f "index.html" ]]; then
    echo -e "${GREEN}[✓]${NC} Opening silicon design website..."
    case "$(uname -s)" in
        Darwin)  open "index.html" 2>/dev/null || true ;;
        Linux)   xdg-open "index.html" 2>/dev/null || true ;;
    esac
fi

# ── Launch Streamlit ────────────────────────────────────────────────────────
echo -e "${GREEN}[✓]${NC} Starting Streamlit dashboard on port ${PORT}..."
echo -e "    Press ${BOLD}Ctrl+C${NC} to stop"
echo ""

export STREAMLIT_PORT="$PORT"
exec python -m streamlit run "$APP_PATH" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
