#!/bin/bash
# virtual environment activation script
# Run with: source path-to/setup-server.sh
# Might cause terminal to go unstable, not yet found out exactly why

set -e

# Absolute path to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Virtual environment directory NEXT TO the script
VENV_DIR="${SCRIPT_DIR}"

# Detect if a virtual environment is already active
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Warning: A virtual environment is already active: $VIRTUAL_ENV"
    echo "Skipping creation/activation to avoid nested venvs."
    return 0 2>/dev/null || exit 0
fi

# Create venv if it doesn't exist
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

# Activate the venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Optional: ensure PATH is correct
export PATH="$VENV_DIR/bin:$PATH"
unset PYTHONHOME

echo "Virtual environment is now active."
echo "Python: $(python --version)"
echo "pip: $(pip --version)"

# === Package installation ===
# Only install packages if a special flag is set (to avoid crashes on login)
if [[ "${INSTALL_PACKAGES:-0}" == "1" ]]; then
    echo "Installing required packages..."
    pip install --upgrade --quiet pip
    pip install --upgrade --quiet ansible-runner ansible-core pyzmq pyvisa numpy scipy
    # add other python packages you might need
fi
