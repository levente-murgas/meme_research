while true; do
    python3 pythonscript.py
    sleep 5
    if [ -f /success.txt ]; then
        echo "Script completed successfully."
        exit 0
    else
        echo "Script stopped prematurely. Restarting..."
        rm /success.txt # Remove the success file if it exists
    fi  
done