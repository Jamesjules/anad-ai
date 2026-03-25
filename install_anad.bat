@echo off
title ANAD — One Click Installer
color 0A

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║           ANAD — ONE CLICK INSTALLER            ║
echo  ║     Public AI — Owned by everyone               ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo  This will install everything Anad needs:
echo    - Python 3.11 (if not installed)
echo    - Git (if not installed)
echo    - All Python packages
echo    - Anad repository
echo    - Correct folder structure
echo.
echo  Keep this window open. Do not close it.
echo.
pause

:: ── Set install directory ────────────────────────────────
set ANAD_DIR=%USERPROFILE%\anad-ai
set LOG=%USERPROFILE%\anad_install_log.txt
echo Install log: %LOG%
echo. > %LOG%

echo.
echo  [1/7] Checking Python...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python not found. Downloading Python 3.11...
    echo  Python not found — downloading >> %LOG%
    
    :: Download Python installer
    powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python_installer.exe'}"
    
    if exist "%TEMP%\python_installer.exe" (
        echo  Installing Python 3.11...
        "%TEMP%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        echo  Python installed. >> %LOG%
        :: Refresh PATH
        call refreshenv 2>nul
        set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts"
    ) else (
        echo.
        echo  ERROR: Could not download Python.
        echo  Please install Python 3.11 manually from:
        echo  https://www.python.org/downloads/
        echo  Make sure to check "Add Python to PATH"
        echo.
        pause
        exit /b 1
    )
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
    echo  Python %PYVER% found. OK
    echo  Python %PYVER% found >> %LOG%
)

echo.
echo  [2/7] Checking Git...
echo.

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Git not found. Downloading Git...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe' -OutFile '%TEMP%\git_installer.exe'}"
    
    if exist "%TEMP%\git_installer.exe" (
        echo  Installing Git...
        "%TEMP%\git_installer.exe" /VERYSILENT /NORESTART /NOCANCEL /SP-
        set "PATH=%PATH%;C:\Program Files\Git\cmd"
        echo  Git installed. >> %LOG%
    ) else (
        echo.
        echo  ERROR: Could not download Git.
        echo  Please install from: https://git-scm.com/download/win
        echo.
        pause
        exit /b 1
    )
) else (
    for /f "tokens=3" %%v in ('git --version') do set GITVER=%%v
    echo  Git %GITVER% found. OK
    echo  Git found >> %LOG%
)

echo.
echo  [3/7] Setting up Anad folder...
echo.

:: Create clean directory
if exist "%ANAD_DIR%" (
    echo  Existing Anad folder found at %ANAD_DIR%
    echo  Updating existing installation...
    cd /d "%ANAD_DIR%"
    git pull --quiet 2>>%LOG%
) else (
    echo  Cloning Anad from GitHub...
    cd /d "%USERPROFILE%"
    git clone https://github.com/Jamesjules/anad-ai.git anad-ai 2>>%LOG%
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: Could not clone repository.
        echo  Check your internet connection.
        echo.
        pause
        exit /b 1
    )
    cd /d "%ANAD_DIR%"
)
echo  Anad folder ready: %ANAD_DIR%

echo.
echo  [4/7] Creating correct folder structure...
echo.

:: Create all required folders
if not exist "model"    mkdir model
if not exist "tokenizer" mkdir tokenizer
if not exist "training\data" mkdir training\data
if not exist "tests"    mkdir tests
if not exist "node"     mkdir node
if not exist "memory"   mkdir memory
if not exist "mobile"   mkdir mobile
if not exist "checkpoints" mkdir checkpoints
if not exist "anad_data" mkdir anad_data

:: Create __init__.py files so Python treats folders as packages
echo. > model\__init__.py
echo. > tokenizer\__init__.py
echo. > training\__init__.py
echo. > node\__init__.py
echo. > memory\__init__.py

:: Move any misplaced root files into correct folders
if exist "config.py"        move /Y config.py        model\config.py        >nul 2>&1
if exist "model.py"         move /Y model.py         model\model.py         >nul 2>&1
if exist "tokenizer.py"     move /Y tokenizer.py     tokenizer\tokenizer.py >nul 2>&1
if exist "test_model.py"    move /Y test_model.py    tests\test_model.py    >nul 2>&1
if exist "test_tokenizer.py" move /Y test_tokenizer.py tests\test_tokenizer.py >nul 2>&1
if exist "test_trainer.py"  move /Y test_trainer.py  tests\test_trainer.py  >nul 2>&1

echo  Folder structure OK.

echo.
echo  [5/7] Installing Python packages...
echo.

:: Upgrade pip silently
python -m pip install --upgrade pip --quiet

:: Core packages
echo  Installing: numpy...
pip install numpy --quiet
echo  Installing: cryptography...
pip install cryptography --quiet
echo  Installing: tqdm aiohttp websockets...
pip install tqdm aiohttp websockets --quiet

:: Detect GPU and install correct PyTorch
echo.
echo  Detecting GPU...
python -c "import subprocess; r=subprocess.run(['nvidia-smi'], capture_output=True); print('NVIDIA' if r.returncode==0 else 'CPU')" > %TEMP%\gpucheck.txt 2>nul
set /p GPUCHECK=<%TEMP%\gpucheck.txt

if "%GPUCHECK%"=="NVIDIA" (
    echo  NVIDIA GPU detected — installing PyTorch with CUDA...
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    echo  PyTorch with CUDA installed >> %LOG%
) else (
    echo  No GPU detected — installing PyTorch CPU version...
    pip install torch --quiet
    echo  PyTorch CPU installed >> %LOG%
)

:: Verify all packages installed
echo.
echo  Verifying packages...
python -c "import torch, numpy, cryptography; print('  All packages OK')"
if %errorlevel% neq 0 (
    echo.
    echo  WARNING: Some packages may not have installed correctly.
    echo  Try running manually: pip install torch numpy cryptography
    echo.
)

echo.
echo  [6/7] Verifying Anad imports...
echo.

python -c "
import sys
sys.path.insert(0, '.')
errors = []
try:
    from model.config import ANAD_NANO
    print('  model.config          OK')
except Exception as e:
    errors.append(f'model.config: {e}')
    print(f'  model.config          MISSING')

try:
    from model.model import AnadModel
    print('  model.model           OK')
except Exception as e:
    errors.append(f'model.model: {e}')
    print(f'  model.model           MISSING')

try:
    from tokenizer.tokenizer import AnadTokenizer
    print('  tokenizer.tokenizer   OK')
except Exception as e:
    errors.append(f'tokenizer: {e}')
    print(f'  tokenizer.tokenizer   MISSING')

try:
    from training.trainer import AnadTrainer
    print('  training.trainer      OK')
except Exception as e:
    errors.append(f'trainer: {e}')
    print(f'  training.trainer      MISSING')

try:
    from training.data_collector import AnadDataCollector
    print('  training.data_collector OK')
except Exception as e:
    errors.append(f'data_collector: {e}')
    print(f'  training.data_collector MISSING')

try:
    from node.identity import AnadIdentity
    print('  node.identity         OK')
except Exception as e:
    errors.append(f'identity: {e}')
    print(f'  node.identity         MISSING')

try:
    from node.node import AnadNode
    print('  node.node             OK')
except Exception as e:
    errors.append(f'node.node: {e}')
    print(f'  node.node             MISSING')

try:
    from memory.memory import AnadMemory
    print('  memory.memory         OK')
except Exception as e:
    errors.append(f'memory: {e}')
    print(f'  memory.memory         MISSING')

try:
    import torch
    print(f'  torch {torch.__version__}          OK')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('  GPU: CPU only')
except Exception as e:
    errors.append(f'torch: {e}')
    print(f'  torch                 MISSING')

if errors:
    print()
    print('  ERRORS:')
    for e in errors:
        print(f'    - {e}')
else:
    print()
    print('  All imports verified.')
"

echo.
echo  [7/7] Creating desktop shortcuts...
echo.

:: Create desktop shortcut for main node
set SHORTCUT_NODE=%USERPROFILE%\Desktop\Anad Node.bat
echo @echo off > "%SHORTCUT_NODE%"
echo title Anad Node >> "%SHORTCUT_NODE%"
echo cd /d "%ANAD_DIR%" >> "%SHORTCUT_NODE%"
echo python main.py >> "%SHORTCUT_NODE%"
echo pause >> "%SHORTCUT_NODE%"

:: Create desktop shortcut for training
set SHORTCUT_TRAIN=%USERPROFILE%\Desktop\Anad Train.bat
echo @echo off > "%SHORTCUT_TRAIN%"
echo title Anad Training >> "%SHORTCUT_TRAIN%"
echo cd /d "%ANAD_DIR%" >> "%SHORTCUT_TRAIN%"
echo python train.py >> "%SHORTCUT_TRAIN%"
echo pause >> "%SHORTCUT_TRAIN%"

:: Create desktop shortcut for update
set SHORTCUT_UPDATE=%USERPROFILE%\Desktop\Anad Update.bat
echo @echo off > "%SHORTCUT_UPDATE%"
echo title Anad Update >> "%SHORTCUT_UPDATE%"
echo cd /d "%ANAD_DIR%" >> "%SHORTCUT_UPDATE%"
echo git pull >> "%SHORTCUT_UPDATE%"
echo pip install -r requirements.txt --quiet >> "%SHORTCUT_UPDATE%"
echo echo Anad updated successfully. >> "%SHORTCUT_UPDATE%"
echo pause >> "%SHORTCUT_UPDATE%"

echo  Desktop shortcuts created:
echo    - "Anad Node.bat"    — start your node
echo    - "Anad Train.bat"   — train the model
echo    - "Anad Update.bat"  — update to latest version

:: ── Done ─────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║           INSTALLATION COMPLETE                 ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo  Anad is installed at:
echo    %ANAD_DIR%
echo.
echo  To start your node:
echo    Double-click "Anad Node.bat" on your desktop
echo    OR run: python main.py
echo.
echo  To start training:
echo    Double-click "Anad Train.bat" on your desktop
echo    OR run: python train.py
echo.
echo  To update Anad:
echo    Double-click "Anad Update.bat" on your desktop
echo.
echo  Your data is stored at:
echo    %ANAD_DIR%\anad_data\
echo    Encrypted. Only you can read it.
echo.
echo  Install log saved at:
echo    %LOG%
echo.

cd /d "%ANAD_DIR%"
echo  Starting Anad node now...
echo.
pause
python main.py
