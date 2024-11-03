#!/bin/bash

VENV_DIR="venv"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created at $VENV_DIR"
fi

    echo "Activating virtual environment..."
# Activate the virtual environment
if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
    source $VENV_DIR/bin/activate
elif [ "$(uname)" == "CYGWIN" ] || [ "$(uname)" == "MINGW" ] || [ "$(uname)" == "MSYS" ]; then
    source $VENV_DIR/Scripts/activate
else
    echo "Unsupported OS"
    exit 1
fi

# Install the required packages
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install required packages"
    exit 1
fi

# Install posix_ipc
pip install posix_ipc
if [ $? -ne 0 ]; then
    echo "Failed to install posix_ipc"
    exit 1
fi

# Deactivate the virtual environment
deactivate

echo "Virtual environment setup complete."