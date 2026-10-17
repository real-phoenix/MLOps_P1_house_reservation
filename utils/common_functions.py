import os
import pandas
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Successfully read YAML file: {file_path}")
            return config
        
    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read YAML file", e)

def load_data(path):
    try:
        logger.info(f"Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error("Error loading the data {e}")
        raise CustomException("Failed to load data", e)
