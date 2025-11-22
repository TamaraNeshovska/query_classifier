import sys
import os
import logging
from datetime import datetime

log_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

# Determine log file path
log_file_name = f"app_{datetime.now().strftime('%Y%m%d')}.log"
project_root = os.environ.get("PROJECT_ROOT") or os.getcwd()  # fallback to cwd
log_dir = os.path.join(project_root, "log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, log_file_name)

# File handler
file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
file_handler.setFormatter(log_format)


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
