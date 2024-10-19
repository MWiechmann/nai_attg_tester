# Standard library imports
import configparser
import ast
import logging
from datetime import datetime
import os
import asyncio
from pathlib import Path

# Third-party imports
from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.Preset import Model
from novelai_api.GlobalSettings import GlobalSettings

def parse_config_value(value):
    """
    Parse a config value, handling both single strings and lists of strings.
    """
    try:
        # Try to evaluate as a Python literal
        parsed_value = ast.literal_eval(value)
        
        # If it's a list, ensure all elements are strings
        if isinstance(parsed_value, list):
            return [str(item) for item in parsed_value]
        elif isinstance(parsed_value, str):
            return parsed_value
        else:
            # If it's neither a list nor a string, return the original value
            return value
    except (ValueError, SyntaxError):
        # If it can't be evaluated, return it as a string
        return value

async def nai_login(api, auth_method, auth):
    """
    Log in to the NovelAI API using the specified authentication method and credentials.
    """
    if auth_method in ["enter_key", "env_key"]:
        await api.high_level.login_from_key(auth)
    elif auth_method in ["enter_token", "env_token"]:
        api.high_level.auth_token = auth  # Directly set the auth_token
    elif auth_method in ["enter_login", "env_login"]:
        await api.high_level.login(auth["user"], auth["pw"])

class ImmediateFileHandler(logging.FileHandler):
    """
    A custom FileHandler that flushes the log file after each write.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(run_name, filename_suffix, project_root):
    """
    Set up logging for the process.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = project_root / 'logs' / f"{run_name}_{filename_suffix}_{timestamp}.log"
    log_filename.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    file_handler = ImmediateFileHandler(str(log_filename))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers in case of multiple calls
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
