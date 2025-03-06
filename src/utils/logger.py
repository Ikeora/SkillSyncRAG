import logging
import os

# Define log file path
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_application.log")

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to a file
        logging.StreamHandler()  # Log to console
    ],
)

def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name)

# Example usage
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logging setup complete!")
