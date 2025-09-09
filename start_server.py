#!/usr/bin/env python3
"""
Simple server startup script for Windows
"""

import os
import sys
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        # Import and run the main server
        from server.main import main
        main()
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
