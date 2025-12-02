#!/bin/bash

# Ensure requirements are installed
pip install -r requirements.txt

# Ensure the log directory exists (optional, but good practice)
mkdir -p UserChatLogs

# Run the doctor bot
python telegram_bot.py
