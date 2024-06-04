# Use the official PyTorch image with GPU support as the base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the hyperparameter tuning script into the container
COPY hyperparameter_tuning.py .

# Set the entrypoint
ENTRYPOINT ["python", "hyperparameter_tuning.py"]