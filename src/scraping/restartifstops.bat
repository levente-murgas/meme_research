@echo off
:loop
python scrape_meme_entries.py
timeout /t 5
if exist "finished.txt" (
    echo Script finished.
    exit /b 0
) 
REM else if exist "banned.txt", start AutoRotateIPIfBanned.exe, delete banned.txt go to loop
if exist "banned.txt" (
    echo IP banned. Restarting...
    del banned.txt
    start AutoRotateIPIfBanned.exe
    timeout /t 20
    goto loop
)
else (
    echo Script stopped prematurely. Restarting...
    goto loop
)
