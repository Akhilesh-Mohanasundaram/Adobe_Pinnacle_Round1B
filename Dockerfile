# Stage 1: Use an official, slim Python runtime as a parent image
# Using a slim image reduces the final size of your Docker image.
# Updated to python:3.10 to support newer package versions like click==8.2.1
FROM python:3.10-slim

# Set the working directory inside the container
# This is where your application code will live.
WORKDIR /app

# Copy the dependencies file first to leverage Docker's layer caching.
# Docker only re-runs this step if requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces the image size by not storing the pip cache.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container.
# This command respects the .dockerignore file.
# All files and folders from your project directory (except those in .dockerignore)
# will be copied into the /app directory inside the container.
COPY . .

# Define the default command to run when the container starts.
# This will execute your main Python script.
CMD ["python", "main_workflow.py"]
