@echo off
setlocal enabledelayedexpansion

set target=%1
shift

:loop
    if "%~1" == "" goto end
    set envname=%~1
    shift

    if "%~1" == "" goto end
    set envvar=%~1
    shift

    set envval=!%envname%!
    if "%envval%" == "" (
        set %envname%=%envvar%
    ) else (
        set %envname%=%envvar%;!%envname%!
    )

    goto loop
:end

%target%

endlocal
exit /b %errorlevel%
