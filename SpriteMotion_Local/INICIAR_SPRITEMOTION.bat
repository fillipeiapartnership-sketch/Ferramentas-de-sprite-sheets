@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title SpriteMotion Local
if exist ".venv\Scripts\python.exe" goto iniciar

where py >nul 2>nul
if errorlevel 1 goto sempython
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.12 -m venv .venv
  goto instalar
)
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.11 -m venv .venv
  goto instalar
)
py -3.13 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.13 -m venv .venv
  goto instalar
)
echo Instale o Python 3.12 x64: https://www.python.org/downloads/release/python-31210/
pause
exit /b 1

:sempython
echo Python Launcher nao encontrado. Instale o Python 3.12 x64.
echo https://www.python.org/downloads/release/python-31210/
pause
exit /b 1

:instalar
if not exist ".venv\Scripts\python.exe" (
  echo Nao foi possivel criar o ambiente Python.
  pause
  exit /b 1
)
echo [SpriteMotion] Instalando dependencias - apenas na primeira execucao...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Falha ao instalar os pacotes. Verifique sua conexao.
  pause
  exit /b 1
)

:iniciar
".venv\Scripts\python.exe" SpriteMotion.py
if errorlevel 1 (
  echo O programa encontrou um erro.
  pause
)
endlocal
