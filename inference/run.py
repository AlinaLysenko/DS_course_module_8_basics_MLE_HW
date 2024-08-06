"""
Script loads the latest trained PyTorch model, data for inference, and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import sys
import torch
import pandas as pd
from datetime import datetime
from typing import List

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Use an environment variable or default to 'settings.json'
CONF_FILE = os.getenv('CONF_PATH', 'settings.json')

from utils import get_project_dir, configure_logging

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initialize parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    latest_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
    return os.path.join(MODEL_DIR, latest_file)


def get_model_by_path(path: str) -> IrisNet:
    """Loads and returns the specified PyTorch model"""
    model = IrisNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    logging.info(f'Model loaded from path: {path}')
    return model


def get_inference_data(path: str):
    """Loads and returns data for inference from the specified csv file"""
    df = pd.read_csv(path)
    encoder = LabelEncoder()
    df['species'] = encoder.fit_transform(df['species'])

    scaler = StandardScaler()
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    x = df.drop('species', axis=1).values
    y = df['species'].values
    return torch.tensor(x, dtype=torch.float32), y


def predict_results(model: IrisNet, infer_data: torch.Tensor) -> List[int]:
    """Predict the results"""
    with torch.no_grad():
        outputs = model(infer_data)
        _, predicted = torch.max(outputs, 1)
    return predicted.tolist()


def store_results(results: List[int], path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results, columns=['Predictions']).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def save_metrics(predictions, labels, path: str = None) -> None:
    """Store the prediction metrics in 'results' directory with current datetime as a filename"""
    if not path:
        path = datetime.now().strftime(conf['general']['datetime_format']) + '_metrics.csv'
        path = os.path.join(RESULTS_DIR, path)
    with open(path, 'w') as f:
        f.write("Accuracy: {}\n".format(accuracy_score(labels, predictions)))
        f.write("Precision: {}\n".format(precision_score(labels, predictions, average='macro')))
        f.write("Recall: {}\n".format(recall_score(labels, predictions, average='macro')))
        f.write("F1-Score: {}\n".format(f1_score(labels, predictions, average='macro')))
    logging.info(f'Metrics saved to {path}')


def main():
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_features, infer_labels = get_inference_data(os.path.join(DATA_DIR, args.infer_file))
    results = predict_results(model, infer_features)
    store_results(results, args.out_path)
    save_metrics(results, infer_labels)
    logging.info('Inference completed successfully.')


if __name__ == "__main__":
    main()
