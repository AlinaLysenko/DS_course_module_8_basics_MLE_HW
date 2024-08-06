# Start from a base Python image
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and data into the container
COPY data/ app/data
COPY data_process/ app/data_process
COPY training/ app/training

# Copy utils.py
COPY utils.py /app
COPY ${CONF_PATH} /app

RUN python data_process/data_loader.py

# Command to run the training script
RUN python, train.py