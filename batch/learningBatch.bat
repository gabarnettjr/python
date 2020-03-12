@echo off

echo The zeroth input is %0, the first is %1, and the second is %2.

setlocal enableextensions enabledelayedexpansion

set /a "x = %1"
set /a "y = %2"
set /a "z = x + y"
set /a "w = x * y"
echo.
echo The sum of %x% and %y% is %z%, and the product is %w%.

echo.

REM a while loop:
set /a "x = 0"
:while1
if %x% leq 5 (
    echo %x%
    set /a "x = x + 1"
    goto :while1
)

echo.

REM another while loop:
:while2
if %x% geq 0 (
	echo %x%
	set /a "x = x - 1"
	goto :while2
)

REM echo.

REM a for loop:
REM The for loop is not working, so commenting it out.
REM for /l %%i in (1,1,5) do (
    REM echo i = %%i
    REM set /a "x = %x% - 1"
    REM echo x = %x%
	REM echo.
REM )

endlocal