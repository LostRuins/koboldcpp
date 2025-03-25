@echo off
setlocal enabledelayedexpansion
echo Create Version File
:: Read the version string from koboldcpp.py
for /f "tokens=2 delims== " %%A in ('findstr "KcppVersion" koboldcpp.py') do (
    set "version=%%~A"
    goto :done
)

:done

:: Display the extracted version (optional, for debugging)
echo Extracted Version: %version%

for /f "tokens=1,2 delims=." %%a in ("%version%") do (
    set version_major=%%a
    set version_minor=%%b
)

echo Major Version: %version_major%
echo Minor Version: %version_minor%

:: Replace all instances of "MYVER" in foo.txt with the version
(
    for /f "delims=" %%i in (version_template.txt) do (
        set "line=%%i"
        set "line=!line:MYVER_MAJOR=%version_major%!"
        set "line=!line:MYVER_MINOR=%version_minor%!"
        echo !line!
    )
) > "version.txt"

endlocal