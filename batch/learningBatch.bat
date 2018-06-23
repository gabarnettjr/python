@echo off

setlocal enableextensions enabledelayedexpansion

set /a "x = 5"
set /a "y = 7"
set /a "z = %x% + %y%"
set /a "w = %x% * %y%"
echo.
echo The sum of %x% and %y% is %z%, and the product is %w%.

echo.

REM a while loop:
set /a "x = 0"
:while1
if %x% leq 5 (
    echo %x%
    set /a "x = %x% + 1"
    goto :while1
)

echo.

REM a for loop:
for /l %%i in (1,1,5) do (
    echo i = %%i
    set /a "x = %x% - 1"
    echo x = %x%
)

endlocal