# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app folder and models folder into the container
COPY app/ ./app/
COPY models/ ./models/

# Expose the port your service will run on
EXPOSE 5000

# Define the command to run your service
CMD ["python", "app/api.py"]