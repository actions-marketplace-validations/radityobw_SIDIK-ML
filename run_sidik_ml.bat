@echo off
title SIDIK-ML - Explainable Non-ANN ML False Positive Reduction
cd /d "%~dp0"

set "PY_CMD=python"
if exist "..\SIDIK\.venv\Scripts\python.exe" set "PY_CMD=..\SIDIK\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"

if "%~1"=="" (
    "%PY_CMD%" -m sidik_ml --gui
    pause
) else (
    "%PY_CMD%" -m sidik_ml %*
)
