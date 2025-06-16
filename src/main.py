"""
AutoSAT Control Panel Entry Point

Main application launcher for the AutoSAT autonomous vehicle control system.
Starts the FastAPI web server with camera streaming, CAN monitoring, and
vehicle control interfaces accessible via web browser.
"""

import logging
import uvicorn

from src.web_interface.app import app

def main():
    """
    Initialize and start the AutoSAT web control panel.
    
    Configures logging, displays startup information, and launches the FastAPI
    server with WebSocket support for real-time vehicle monitoring and control.
    
    Server Configuration:
    - Host: 0.0.0.0 (accessible from any network interface)
    - Port: 8000
    - Features: Camera streaming, CAN data monitoring, vehicle control
    """
    # Configure logging for application monitoring
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting AutoSAT Control Panel")
    logging.info("Web interface will be available at http://localhost:8000")
    logging.info("Features: Camera streaming, CAN monitoring, vehicle control")

    # Launch FastAPI server with uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,       # Standard development port
        log_level="info" # Match application logging level
    )

if __name__ == "__main__":
    """
    Application entry point when run directly.
    
    Usage:
        python -m src.main
        
    Access the control panel at:
        http://localhost:8000 (local access)
        http://<your-ip>:8000 (network access)
    """
    main()