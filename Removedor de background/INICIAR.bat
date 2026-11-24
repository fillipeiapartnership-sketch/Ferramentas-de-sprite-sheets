@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title FundoZero - IA local

echo.
echo ======================================================
echo    FUNDOZERO - REMOCAO LOCAL DE BACKGROUND
 echo ======================================================
echo.

set "PYTHON_EXE="
for %%V in (3.13 3.12 3.11) do (
    if not defined PYTHON_EXE (
        for /f "usebackq delims=" %%P in (`py -%%V -c "import sys;print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
    )
)
if not defined PYTHON_EXE (
    for /f "usebackq delims=" %%P in (`python -c "import sys;assert (3,11) ^<= sys.version_info[:2] ^< (3,14);print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"

if not defined PYTHON_EXE (
    echo O Python 3.11 a 3.13 nao foi localizado.
    echo.
    where winget >nul 2>&1
    if errorlevel 1 goto instrucoes_python
    choice /c SN /m "Deseja instalar Python 3.12 automaticamente via winget?"
    if errorlevel 2 goto instrucoes_python
    echo.
    echo Instalando Python para o seu usuario...
    winget install --id Python.Python.3.12 -e --source winget --scope user --accept-source-agreements --accept-package-agreements
    if errorlevel 1 goto instrucoes_python
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if not defined PYTHON_EXE goto instrucoes_python
)

"%PYTHON_EXE%" -c "import sys,struct;assert (3,11) <= sys.version_info[:2] < (3,14) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python deve ter versao 3.11-3.13, 64 bits.
    goto falha
)

echo Python encontrado: %PYTHON_EXE%
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Preparando o ambiente virtual...
    "%PYTHON_EXE%" -m venv ".venv"
    if errorlevel 1 goto falha
)

".venv\Scripts\python.exe" -c "import importlib.util,sys;sys.exit(0 if all(importlib.util.find_spec(m) for m in ('rembg','PIL','onnxruntime','tkinterdnd2','imageio_ffmpeg')) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Instalando as bibliotecas locais. Internet necessaria apenas nesta etapa.
    ".venv\Scripts\python.exe" -m pip install --no-cache-dir --disable-pip-version-check -r "requirements.txt"
    if errorlevel 1 goto falha
)

echo.
echo Abrindo FundoZero...
".venv\Scripts\python.exe" "app.py"
if errorlevel 1 goto falha
exit /b 0

:instrucoes_python
echo.
echo Para instalar manualmente, visite: https://www.python.org/downloads/
echo Escolha Python 3.12 64 bits e execute este INICIAR.bat novamente.
start "" "https://www.python.org/downloads/"
:falha
echo.
echo Instalacao ou execucao interrompida. Verifique as mensagens acima.
echo Dica: consulte LEIA-ME.txt
pause
exit /b 1
