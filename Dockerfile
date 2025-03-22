# Use a base image with Python installed (official Python image)
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run Streamlit app when the container starts
CMD ["streamlit", "run", "app.py"]
