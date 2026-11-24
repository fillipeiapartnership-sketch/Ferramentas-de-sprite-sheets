@echo off
setlocal
cd /d "%~dp0"
title SpriteMorph Studio - Inicializador
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel% neq 0 (
    echo Python nao encontrado. Instale Python 3.10 ou mais recente em https://www.python.org/downloads/
    pause
    exit /b 1
  )
  set "PY=python"
)
%PY% -c "import PIL,numpy" >nul 2>nul
if %errorlevel% neq 0 (
  echo Instalando bibliotecas necessarias... ^(requer internet apenas na primeira instalacao^)
  %PY% -m pip install -r requirements.txt
  if %errorlevel% neq 0 (
    echo Falha ao instalar dependencias.
    pause
    exit /b 1
  )
)
%PY% app.py
if %errorlevel% neq 0 pause
