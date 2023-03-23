@echo off
:loop
python scrape_meme_entries.py
timeout /t 5
if exist "success.txt" (
    echo Script completed successfully.
    exit /b 0
) else (
    echo Script stopped prematurely. Restarting...
    goto loop
)
