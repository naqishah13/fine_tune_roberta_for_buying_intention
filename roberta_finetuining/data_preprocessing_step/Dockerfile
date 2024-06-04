# Use the official Python image from the Docker Hub
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data processing script into the container
COPY data_processing.py .

# Set the entrypoint
ENTRYPOINT ["python", "data_processing.py"]
