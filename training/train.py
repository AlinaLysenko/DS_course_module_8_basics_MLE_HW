"""
This script prepares the data, runs the training, and saves the model.
"""
import logging
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

logger.info("Defining paths...")

# Configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['data_dir'])))
DATA_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['models_dir'])))
MODEL_PATH = os.path.join(MODEL_DIR, conf['inference']['model_name'])
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logger.info(f"Directory  {MODEL_DIR} was created.")
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 50


# Model Definition
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


# Load data
def load_data():
    df = pd.read_csv(DATA_PATH)

    encoder = LabelEncoder()
    df['species'] = encoder.fit_transform(df['species'])

    scaler = StandardScaler()
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    x = df.drop('species', axis=1).values
    y = df['species'].values
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Train model
def train_model():
    x_train, y_train = load_data()
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved to ", MODEL_PATH)


if __name__ == "__main__":
    train_model()
