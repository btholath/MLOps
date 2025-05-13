import logging
import os
from datetime import datetime

# Create timestamped log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the log directory path
log_dir = os.path.join(os.getcwd(), "logs")

# Create log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Basic logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s'
)

# Example log message
logging.info("Logging system initialized successfully.")
