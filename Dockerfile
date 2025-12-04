# Use an official Python runtime as a parent image
# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker# you will also find guides on how best to write your Dockerfile
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# IMPORTANT: Ensure your vector_db folder is committed to GitHub
COPY . .

# Grant execution permissions to the run script
RUN chmod +x run.sh

# This tells the container what command to run when it starts
ENTRYPOINT ["./run.sh"]
