#!/usr/bin/env python
"""
Run script for the Mental Health Prediction API
"""
import uvicorn
import os
import sys

def main():
    """Run the FastAPI application"""
    print("Starting Mental Health Prediction API...")
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Check if running in debug mode
    debug_mode = "--debug" in sys.argv
    
    if debug_mode:
        print("Running in debug mode with auto-reload enabled")
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=debug_mode
    )

if __name__ == "__main__":
    main() 