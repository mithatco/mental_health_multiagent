#!/usr/bin/env python3
"""
Chat Log Viewer

This script allows viewing saved chat logs from the mental health multi-agent system.
"""

import os
import sys
import argparse
from utils.chat_logger import ChatLogger

def display_chat(chat_data):
    """Display a chat conversation in a readable format."""
    print("=" * 60)
    print(f"Conversation using questionnaire: {chat_data['questionnaire']}")
    print(f"Time: {chat_data['timestamp']}")
    print("=" * 60)
    
    for msg in chat_data['conversation']:
        role = msg['role'].upper()
        if role == "SYSTEM":
            continue  # Skip system messages
        print(f"\n{role}:")
        print(f"{msg['content']}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    print(diagnosis := chat_data['diagnosis'])
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Chat Log Viewer")
    parser.add_argument('--list', action='store_true', help="List available chat logs")
    parser.add_argument('--log', type=str, help="Specify a log file to view")
    parser.add_argument('--logs-dir', type=str, default="chat_logs", help="Directory containing chat logs (default: chat_logs)")
    args = parser.parse_args()
    
    # Initialize the chat logger
    chat_logger = ChatLogger(args.logs_dir)
    
    if args.list:
        # List available logs
        logs = chat_logger.list_chat_logs()
        json_logs = [log for log in logs if log.endswith('.json')]
        
        if not json_logs:
            print("No chat logs found.")
            return
        
        print(f"Found {len(json_logs)} chat logs:")
        for i, log in enumerate(sorted(json_logs), 1):
            print(f"{i}. {log}")
        
        try:
            choice = int(input("\nEnter number to view (0 to exit): "))
            if choice == 0:
                return
            if 1 <= choice <= len(json_logs):
                selected_log = sorted(json_logs)[choice-1]
                chat_data = chat_logger.load_chat(selected_log)
                display_chat(chat_data)
            else:
                print("Invalid selection.")
        except (ValueError, IndexError):
            print("Invalid selection.")
    
    elif args.log:
        # View a specific log
        try:
            chat_data = chat_logger.load_chat(args.log)
            display_chat(chat_data)
        except Exception as e:
            print(f"Error loading chat log: {str(e)}")
    
    else:
        # No options specified, list logs by default
        print("Available options:")
        print("  --list      List available chat logs")
        print("  --log FILE  View a specific chat log")
        print("\nExample: python view_chats.py --list")

if __name__ == "__main__":
    main()
