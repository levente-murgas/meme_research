@echo off
:loop
python scrape_meme_entries.py
timeout /t 5
if exist "finished.txt" (
    echo Script finished.
    exit /b 0
) else (
    echo Script stopped prematurely. Restarting...
    goto loop
)
