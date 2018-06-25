@echo off

setlocal enableextensions enabledelayedexpansion

for /R . %%f in (*) do (
    set B=%%f
    echo Relative !B:%CD%\=!
)

REM for /R %cd%/dir1 %%d in (.) do (
    REM if exist %%d\* echo %%d
    REM if not exist %%d\* type %%d
REM )

endlocal