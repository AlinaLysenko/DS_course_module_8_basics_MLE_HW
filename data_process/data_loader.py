# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
import json

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['data_dir'])))
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Method to load and split Iris dataset
def load_and_split_data():
    logger.info("Loading and preparing Iris dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(url, names=column_names)

    train, infer = train_test_split(df, test_size=0.2, random_state=42)
    logger.info("Data split into training and inference datasets.")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Directory  {DATA_DIR} was created.")
    train.to_csv(TRAIN_PATH, index=False)
    infer.to_csv(INFERENCE_PATH, index=False)
    logger.info(f"Training data saved to {TRAIN_PATH}")
    logger.info(f"Inference data saved to {INFERENCE_PATH}")


# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting script...")
    load_and_split_data()
    logger.info("Script completed successfully.")
