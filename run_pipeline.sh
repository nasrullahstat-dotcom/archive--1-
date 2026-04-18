#!/bin/bash

# Respiratory Disease Detection Pipeline - Quick Start
# For Mac/Linux

echo ""
echo "========================================"
echo "Respiratory Disease Detection Pipeline"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found! Please install Python 3.8+"
    exit 1
fi

echo "[1/4] Checking Python installation... OK"
python3 --version
echo ""

# Check/create virtual environment
if [ ! -d "env" ]; then
    echo "[2/4] Creating virtual environment..."
    python3 -m venv env
    echo "Virtual environment created!"
else
    echo "[2/4] Virtual environment found"
fi

# Activate environment
echo "Activating environment..."
source env/bin/activate
echo ""

# Install requirements
echo "[3/4] Installing dependencies..."
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed!"
echo ""

# Menu
echo "[4/4] Select what to run:"
echo ""
echo "1. Run Data Preprocessing (01_data_preprocessing.py)"
echo "2. Train Models (02_model_training.py)"
echo "3. Launch Web App (03_streamlit_app.py)"
echo "4. Run Full Pipeline (1, 2, 3)"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Starting Data Preprocessing..."
        echo "This may take 30-45 minutes depending on your CPU."
        echo ""
        python 01_data_preprocessing.py
        ;;
    2)
        echo ""
        echo "Starting Model Training..."
        echo "This may take 5-10 minutes."
        echo ""
        python 02_model_training.py
        ;;
    3)
        echo ""
        echo "Starting Streamlit Web App..."
        echo "Opening browser at http://localhost:8501"
        echo ""
        streamlit run 03_streamlit_app.py
        ;;
    4)
        echo ""
        echo "Running full pipeline..."
        echo "Total estimated time: 40-60 minutes"
        echo ""
        
        echo "===== STEP 1: Data Preprocessing ====="
        python 01_data_preprocessing.py
        if [ $? -ne 0 ]; then
            echo "ERROR in preprocessing!"
            exit 1
        fi
        echo ""
        
        echo "===== STEP 2: Model Training ====="
        python 02_model_training.py
        if [ $? -ne 0 ]; then
            echo "ERROR in training!"
            exit 1
        fi
        echo ""
        
        echo "===== STEP 3: Launching Web App ====="
        streamlit run 03_streamlit_app.py
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
