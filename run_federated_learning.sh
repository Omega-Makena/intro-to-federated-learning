#!/bin/bash

# Script to run federated learning simulation
# This script starts the server and multiple clients in the background

echo "========================================"
echo "Federated Learning - Credit Fraud Detection"
echo "========================================"
echo ""

# Configuration
NUM_CLIENTS=3
NUM_ROUNDS=10
LOCAL_EPOCHS=5
SERVER_ADDRESS="127.0.0.1:8080"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment should be used
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import flwr" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting federated learning simulation..."
echo "Configuration:"
echo "  - Number of clients: $NUM_CLIENTS"
echo "  - Number of rounds: $NUM_ROUNDS"
echo "  - Local epochs: $LOCAL_EPOCHS"
echo ""

# Start server in background
echo "Starting server..."
python src/server.py \
    --num-rounds $NUM_ROUNDS \
    --min-clients $NUM_CLIENTS \
    --local-epochs $LOCAL_EPOCHS \
    --server-address $SERVER_ADDRESS &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to start
sleep 3

# Start clients in background
echo ""
echo "Starting clients..."
for i in $(seq 0 $(($NUM_CLIENTS - 1))); do
    python src/client.py \
        --client-id $i \
        --server-address $SERVER_ADDRESS \
        --num-clients $NUM_CLIENTS \
        --data-split iid &
    
    CLIENT_PID=$!
    echo "Client $i started (PID: $CLIENT_PID)"
    sleep 1
done

echo ""
echo "All clients started. Training in progress..."
echo "Press Ctrl+C to stop the simulation."
echo ""

# Wait for server to complete
wait $SERVER_PID

echo ""
echo "========================================"
echo "Federated Learning Completed!"
echo "========================================"
