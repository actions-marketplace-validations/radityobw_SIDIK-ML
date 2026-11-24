@echo off
chcp 65001 >nul
title SIDIK-ML - Interactive Non-ANN ML Console
cd /d "%~dp0"

set "PY_CMD=python"
if exist "..\SIDIK\.venv\Scripts\python.exe" set "PY_CMD=..\SIDIK\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"

if "%~1"=="" (
    "%PY_CMD%" -m sidik_ml -i
) else (
    "%PY_CMD%" -m sidik_ml %*
    pause
)
