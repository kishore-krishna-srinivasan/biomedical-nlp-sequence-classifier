#!/bin/bash

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install pytest

# Start the FastAPI server in the background
uvicorn app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for the server to start
sleep 5

# Run tests
pytest


