import logging
import uvicorn

from src.web_interface.app import app

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting AutoSAT Control Panel")

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()