@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ================================================
echo      J.A.R.V.I.S. COMPANION BUILDER
echo ================================================
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python launcher ^(py^) was not found.
    echo Install Python 3.11 or newer, then run this builder again.
    pause
    exit /b 1
)

if not exist ".buildvenv\Scripts\python.exe" (
    echo Creating build environment...
    py -3 -m venv .buildvenv
    if errorlevel 1 goto :fail
)

call ".buildvenv\Scripts\activate.bat"
if errorlevel 1 goto :fail

python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist "local_responses.py" (
    echo ERROR: local_responses.py is missing.
    goto :fail
)
if not exist "edith_gemini_voice.py" (
    echo ERROR: edith_gemini_voice.py is missing.
    goto :fail
)
if not exist "JARVIS-Companion.iss" (
    echo ERROR: JARVIS-Companion.iss is missing.
    goto :fail
)

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist output rmdir /s /q output
mkdir output

echo.
echo Building J.A.R.V.I.S. Companion...
echo.
python -m PyInstaller --noconfirm --clean --onedir --noconsole --name "JARVIS Companion" ^
  --hidden-import=local_responses ^
  --hidden-import=google.genai ^
  --hidden-import=google.genai.types ^
  --hidden-import=google.genai.models ^
  --hidden-import=openwakeword ^
  --hidden-import=onnxruntime ^
  --collect-all=google.genai ^
  --collect-all=openwakeword ^
  --collect-all=onnxruntime ^
  --collect-all=edge_tts ^
  --collect-all=pygame ^
  --collect-all=sounddevice ^
  --collect-all=soundfile ^
  edith_gemini_voice.py
if errorlevel 1 goto :fail

where ISCC.exe >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyInstaller build completed, but Inno Setup compiler ISCC.exe was not found.
    echo Install Inno Setup 6 and make sure ISCC.exe is on PATH, then rerun this builder.
    echo.
    echo The unpacked application is here:
    echo   %cd%\dist\JARVIS Companion\
    pause
    exit /b 2
)

echo.
echo Building Windows installer...
echo.
ISCC.exe "JARVIS-Companion.iss"
if errorlevel 1 goto :fail

echo.
echo ================================================
echo BUILD COMPLETE
echo ================================================
echo Installer:
echo %cd%\output\JARVIS-Setup.exe
echo.
pause
exit /b 0

:fail
echo.
echo ================================================
echo BUILD FAILED
echo ================================================
echo Check the error above.
pause
exit /b 1
