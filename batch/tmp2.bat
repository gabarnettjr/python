@echo off

set self=%0
set source=%1
set destination=%2

setlocal enableextensions enabledelayedexpansion

for /R %%f in (*) do (
    set B=%%f
    echo !B!
    echo !B:%CD%\=!
    echo !B:C:\cygwin64\=!
    echo.
    REM echo %destination%\!B:%CD%\=!
    REM echo %source%\!B:%CD%\=!
    REM mklink %destination%\!B:%CD%\=! %source%\!B:%CD%\=!
)

endlocal
