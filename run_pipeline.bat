@echo off
REM Respiratory Disease Detection Pipeline - Quick Start
REM For Windows

echo.
echo ========================================
echo Respiratory Disease Detection Pipeline
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo [1/4] Checking Python installation... OK
echo.

REM Check virtual environment
if not exist "env" (
    echo [2/4] Creating virtual environment...
    python -m venv env
    call env\Scripts\activate.bat
    echo Virtual environment created!
) else (
    echo [2/4] Activating virtual environment...
    call env\Scripts\activate.bat
)
echo.

REM Install requirements
echo [3/4] Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed!
echo.

REM Ask what to run
echo [4/4] Select what to run:
echo.
echo 1. Run Data Preprocessing (01_data_preprocessing.py)
echo 2. Train Models (02_model_training.py)
echo 3. Launch Web App (03_streamlit_app.py)
echo 4. Run Full Pipeline (1, 2, 3)
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting Data Preprocessing...
    echo This may take 30-45 minutes depending on your CPU.
    echo.
    python 01_data_preprocessing.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Model Training...
    echo This may take 5-10 minutes.
    echo.
    python 02_model_training.py
) else if "%choice%"=="3" (
    echo.
    echo Starting Streamlit Web App...
    echo Opening browser at http://localhost:8501
    echo.
    streamlit run 03_streamlit_app.py
) else if "%choice%"=="4" (
    echo.
    echo Running full pipeline...
    echo Total estimated time: 40-60 minutes
    echo.
    
    echo ===== STEP 1: Data Preprocessing =====
    python 01_data_preprocessing.py
    if errorlevel 1 (
        echo ERROR in preprocessing!
        pause
        exit /b 1
    )
    echo.
    
    echo ===== STEP 2: Model Training =====
    python 02_model_training.py
    if errorlevel 1 (
        echo ERROR in training!
        pause
        exit /b 1
    )
    echo.
    
    echo ===== STEP 3: Launching Web App =====
    streamlit run 03_streamlit_app.py
) else (
    echo Exiting...
    exit /b 0
)

pause
