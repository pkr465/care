"""
launch_streamlit.py

Please run from the project root; do NOT run from inside ui/.
The default Streamlit port is 8502; change here or in .env if needed.
"""

import os
import sys
import subprocess
import socket

def get_local_ip():
    ip = "localhost"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        pass
    return ip

def print_dashboard_access_info(port):
    local_ip = "localhost"
    net_ip = get_local_ip()
    print("\nHow to access this dashboard:\n")
    print(f"On this machine: http://{local_ip}:{port}")
    print(f"On another device on the same network: http://{net_ip}:{port}")
    print('Note: "0.0.0.0" is a server listening address—not a real URL.')
    print("Always use 'localhost' or your computer's network IP as above.\n")

def main():
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required for Streamlit.", file=sys.stderr)
        sys.exit(1)

    # Check that we are being run from the project root and that ui/ exists
    if not os.path.isdir("ui"):
        print("ERROR: Please run this script from the project root directory (not from within ui/). Folder 'ui/' not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile("ui/streamlit_app.py"):
        print("ERROR: File 'ui/streamlit_app.py' not found in the ui/ directory.", file=sys.stderr)
        sys.exit(1)

    # Check Streamlit installation
    try:
        import streamlit
    except ImportError:
        print("ERROR: Streamlit is not installed. Please install it with 'pip install streamlit'.", file=sys.stderr)
        sys.exit(1)

    # Get port from environment or default
    port = os.environ.get("STREAMLIT_PORT", "8502")
    print_dashboard_access_info(port)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port", str(port)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Streamlit failed to launch (exit code {e.returncode}).", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()